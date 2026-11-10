"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import defaultdict, deque
import math

def construct():
    p = 101
    # Precompute modular inverses for denominators 1..10 mod 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Generate all non-empty subsets of {1..10} and their inverse sums
    # subset represented as bitmask, size = popcount
    subsets_by_size = [[] for _ in range(11)]  # index by size
    all_subsets = []  # (mask, size, residue)
    
    for mask in range(1, 1 << 10):
        terms = []
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                terms.append(x)
        residue = sum(inv[x] for x in terms) % p
        size = len(terms)
        subsets_by_size[size].append((mask, residue))
        all_subsets.append((mask, size, residue))
    
    # For each residue, collect which subset sizes can represent it
    # We want to find minimum k such that we can assign to each residue r
    # a subset of size <= k, with distinct subsets (each subset used at most once)
    # This is a bipartite matching: left = 101 residues, right = subsets
    # Edge exists if subset's residue == r and subset size <= k
    
    # Binary search on k (max size)
    # For a given k, build bipartite graph and check if perfect matching exists
    
    def hopcroft_karp(n_left, n_right, adj):
        """
        n_left: number of left nodes (0..n_left-1)
        n_right: number of right nodes (0..n_right-1)
        adj: list of lists, adj[i] = list of right nodes connected to left node i
        Returns matching size
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching
    
    # For each k from 1 to 10, check if perfect matching exists
    # using only subsets of size <= k
    
    # Precompute for each k, the adjacency for each residue
    # adj_for_k[k][residue] = list of subset indices (global index in all_subsets)
    # But we need to be careful: each subset can only be used once
    
    # Let's index subsets by their position in all_subsets
    # For a given k, right side = all subsets with size <= k
    # We need to map: for each residue r, which subset indices (in the filtered set) have residue r
    
    # Precompute: for each k, for each residue, list of global subset indices
    subset_indices_by_size = [[] for _ in range(11)]
    for i, (mask, size, residue) in enumerate(all_subsets):
        subset_indices_by_size[size].append(i)
    
    # For binary search, try k = 1, 2, ..., 10
    # Since the answer is likely small, try from 1 upward
    
    best_k = 10
    best_matching = None
    
    for k in range(1, 11):
        # Right side: all subsets with size <= k
        right_indices = []
        for s in range(1, k + 1):
            right_indices.extend(subset_indices_by_size[s])
        
        n_right = len(right_indices)
        if n_right < 101:
            continue
        
        # Map global subset index to local right index
        global_to_local = {}
        for local_idx, global_idx in enumerate(right_indices):
            global_to_local[global_idx] = local_idx
        
        # Build adjacency: for each residue r (0..100), list of local right indices
        adj = [[] for _ in range(101)]
        for global_idx in right_indices:
            mask, size, residue = all_subsets[global_idx]
            if size <= k:
                local_idx = global_to_local[global_idx]
                adj[residue].append(local_idx)
        
        # Check if every residue has at least one subset
        valid = True
        for r in range(101):
            if not adj[r]:
                valid = False
                break
        if not valid:
            continue
        
        # Run Hopcroft-Karp
        matching_size = hopcroft_karp(101, n_right, adj)
        
        if matching_size == 101:
            best_k = k
            best_matching = (k, right_indices, global_to_local, adj)
            break
    
    if best_matching is None:
        # Fallback: use the original greedy approach
        representations = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            residue = sum(inv[x] for x in terms) % p
            representations.setdefault(residue, terms)
        return [representations[r] for r in range(101)]
    
    k, right_indices, global_to_local, adj = best_matching
    
    # Re-run Hopcroft-Karp to get the actual matching
    n_right = len(right_indices)
    pair_u = [-1] * 101
    pair_v = [-1] * n_right
    dist = [0] * 101
    
    def bfs():
        queue = deque()
        for u in range(101):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    while bfs():
        for u in range(101):
            if pair_u[u] == -1:
                dfs(u)
    
    # Extract result
    result = []
    for r in range(101):
        local_idx = pair_u[r]
        if local_idx == -1:
            # Should not happen if matching is perfect
            # Fallback
            terms = [x for x in range(1, 11) if all_subsets[right_indices[0]][0] & (1 << (x - 1))]
            result.append(terms)
        else:
            global_idx = right_indices[local_idx]
            mask = all_subsets[global_idx][0]
            terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
            result.append(terms)
    
    return result
# EVOLVE-BLOCK-END
