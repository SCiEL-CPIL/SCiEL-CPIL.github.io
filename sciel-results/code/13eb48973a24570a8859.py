import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Q3 structure: 3-cube with 8 vertices, 12 edges
    # Vertices are 3-bit numbers 0..7
    # Edge if Hamming distance 1
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))
    
    # Precompute Q3 adjacency as bitmasks for fast checking
    q3_adj_mask = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj_mask[u] |= (1 << v)
        q3_adj_mask[v] |= (1 << u)
        
    # Precompute neighbors in Q3 for each vertex
    q3_neighbors = [[] for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)

    def creates_q3(adj_mask, u, v):
        """
        Check if adding edge (u, v) to a Q3-free graph creates a Q3.
        Assumes u and v are adjacent in adj_mask.
        Returns True if a Q3 containing (u, v) exists.
        """
        # Try all orientations of Q3 edges matching (u, v)
        for qa, qb in Q3_EDGES:
            # Case 1: qa -> u, qb -> v
            # Case 2: qa -> v, qb -> u
            
            for orientation in [(u, v), (v, u)]:
                map_qa, map_qb = orientation
                # Fixed mapping
                mapping = [-1] * 8
                mapping[qa] = map_qa
                mapping[qb] = map_qb
                
                # Remaining vertices in Q3
                remaining_q3 = [x for x in range(8) if x != qa and x != qb]
                # Remaining vertices in G
                # We can just pick from 0..15 excluding u, v
                
                # Order remaining_q3 by constraints? 
                # All have degree 3. 
                # Let's just iterate.
                
                def dfs(idx):
                    if time.time() > deadline - 0.2:
                        return False
                    if idx == len(remaining_q3):
                        return True
                    
                    q3_vertex = remaining_q3[idx]
                    # Neighbors of q3_vertex in Q3 that are already mapped
                    mapped_neighbors = []
                    for nb in q3_neighbors[q3_vertex]:
                        if mapping[nb] != -1:
                            mapped_neighbors.append(mapping[nb])
                    
                    # Try all unused vertices in G
                    # We need to ensure we don't pick u or v (already used)
                    # And we need to ensure we don't pick a vertex already mapped to another q3 vertex
                    used_mask = 0
                    for m in mapping:
                        if m != -1:
                            used_mask |= (1 << m)
                            
                    for g_vertex in range(n):
                        if (used_mask >> g_vertex) & 1:
                            continue
                        
                        # Check if g_vertex is adjacent to all mapped neighbors
                        valid = True
                        for nb in mapped_neighbors:
                            if not (adj_mask[nb] >> g_vertex) & 1:
                                valid = False
                                break
                        
                        if valid:
                            mapping[q3_vertex] = g_vertex
                            if dfs(idx + 1):
                                return True
                            mapping[q3_vertex] = -1
                            
                    return False

                if dfs(0):
                    return True
        
        return False

    def find_one_q3_fast(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        adj_mask: list of 16 integers, each a bitmask of neighbors.
        """
        # Iterate over all 8-element subsets
        all_8_subsets = list(itertools.combinations(range(n), 8))
        for S in all_8_subsets:
            S_set = set(S)
            
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                mask = adj_mask[v] & ((1 << 16) - 1)
                deg = 0
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Backtracking with bitmask pruning
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S
            
            def dfs(idx):
                nonlocal used
                if time.time() > deadline - 0.5:
                    return False  # Abort if too slow
                
                if idx == 8:
                    return True
                
                # Neighbors of idx in Q3 that are already assigned (index < idx)
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                # Try each unused vertex in S
                for v in S:
                    if (used >> v) & 1:
                        continue
                    
                    # Check if v is adjacent to all already-assigned neighbors of idx
                    valid = True
                    for nb in assigned_nbs:
                        if not (adj_mask[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                
                return False
            
            if dfs(0):
                # Extract the 12 edges of the found Q3
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return (True, edges)
        
        return (False, None)
    
    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    # Strategy: Large Neighborhood Search (LNS)
    # 1. Start with K_16
    # 2. Remove edges to make Q3-free
    # 3. Local search to add edges using creates_q3
    # 4. Perturb and repeat
    
    best_m = 0
    best_adj = [0] * n
    best_edges = []
    
    # Initial solution: K_16
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    current_edges = _count_edges(adj)
    
    # Phase 1: Remove edges to eliminate Q3
    while time.time() < deadline - 10.0:
        found, edges = find_one_q3_fast(adj)
        if not found:
            break
        
        # Remove a random edge from the found Q3
        random.shuffle(edges)
        u, v = edges[0]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_edges -= 1
    
    # Phase 2: Local Search to add edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                candidates.append((u, v))
    
    for restart in range(100): # Up to 100 restarts
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(candidates)
        improved = False
        
        for u, v in candidates:
            if time.time() > deadline - 2.0:
                break
                
            # Check if edge is already present
            if (adj[u] >> v) & 1:
                continue
                
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            
            # Check if this creates a Q3
            if creates_q3(adj, u, v):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                # Success
                improved = True
                # Update best if better
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                    best_edges = get_edge_list(adj)
                    
        # If no improvement, perturb
        if not improved and current_edges > 0:
            existing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        existing_edges.append((u, v))
            
            if existing_edges:
                u, v = random.choice(existing_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                candidates.append((u, v))
    
    # Final update
    if current_edges > best_m:
        found, _ = find_one_q3_fast(adj)
        if not found:
            best_m = current_edges
            best_adj = adj[:]
            best_edges = get_edge_list(adj)
            
    return best_edges
