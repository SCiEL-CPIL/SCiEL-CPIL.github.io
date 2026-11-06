"""Optimized construct: minimum-cardinality assignment per residue via size-ordered search."""
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
    
    denominators = list(range(1, 11))
    
    # Precompute residue for every subset (by mask)
    # mask: bits 0..9 correspond to denominators 1..10
    subset_residue = [0] * (1 << 10)
    subset_size = [0] * (1 << 10)
    for mask in range(1, 1 << 10):
        # Use lowest set bit to build up
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-indexed, corresponds to denominator idx+1
        prev = mask ^ lb
        subset_residue[mask] = (subset_residue[prev] + inv[idx + 1]) % p
        subset_size[mask] = subset_size[prev] + 1
    
    # Group masks by size
    by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        by_size[subset_size[mask]].append(mask)
    
    # Hopcroft-Karp implementation
    def hopcroft_karp(left_nodes, right_nodes, adjacency):
        """
        left_nodes: list of left node indices (residues)
        right_nodes: list of right node indices (subset masks)
        adjacency: dict mapping left_node -> list of right_nodes
        Returns: list of (left_node, right_node) pairs
        """
        # Map right nodes to indices for pairU/V arrays
        right_index_map = {r: i for i, r in enumerate(right_nodes)}
        
        pairU = {u: -1 for u in left_nodes}
        pairV = {v: -1 for v in right_nodes}
        
        def bfs():
            from collections import deque
            dist = {u: 0 for u in left_nodes}
            queue = deque()
            for u in left_nodes:
                if pairU[u] == -1:
                    queue.append(u)
                else:
                    dist[u] = -1
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adjacency.get(u, []):
                    if v not in right_index_map:
                        continue
                    pu = pairV[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adjacency.get(u, []):
                if v not in right_index_map:
                    continue
                pu = pairV[v]
                if pu == -1 or (dist.get(pu, -1) == dist.get(u, -1) + 1 and dfs(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = []
        while bfs():
            for u in left_nodes:
                if pairU[u] == -1:
                    if dfs(u):
                        matching.append((u, pairU[u]))
        
        return matching

    # Assign minimum-cardinality subset to each residue
    assigned = [None] * p  # assigned[r] = list of denominators
    assigned_mask = [None] * p  # assigned_mask[r] = mask
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        # Find unassigned residues
        unassigned_residues = [r for r in range(p) if assigned[r] is None]
        if not unassigned_residues:
            break
            
        # Find subsets of size k that map to unassigned residues
        # We can restrict right nodes to those subsets that map to an unassigned residue
        potential_subsets = []
        adj = {r: [] for r in unassigned_residues}
        
        for mask in by_size[k]:
            r = subset_residue[mask]
            if assigned[r] is None:
                potential_subsets.append(mask)
                adj[r].append(mask)
        
        if not potential_subsets:
            continue
            
        # Run Hopcroft-Karp
        # Left: unassigned_residues
        # Right: potential_subsets
        matches = hopcroft_karp(unassigned_residues, potential_subsets, adj)
        
        # Apply matches
        for r, mask in matches:
            terms = [i + 1 for i in range(10) if mask >> i & 1]
            assigned[r] = terms
            assigned_mask[r] = mask
    
    # If any residue is still unassigned (shouldn't happen, but safety), fill with full set
    full_set = list(range(1, 11))
    full_mask = (1 << 10) - 1
    for r in range(p):
        if assigned[r] is None:
            assigned[r] = full_set
            assigned_mask[r] = full_mask
    
    return [assigned[r] for r in range(p)]
