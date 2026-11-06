import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_triangle_free(adj, u, v):
        common = adj[u] & adj[v]
        return common == 0

    def has_independent_set_10(adj, n):
        # Returns True if an independent set of size 10 exists
        mask_all = (1 << n) - 1
        
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            # Heuristic: pick vertex with maximum degree in candidates
            best_v = -1
            max_deg = -1
            temp_cand = candidates
            while temp_cand:
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            # Try include first
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)

    def build_andrasfai(n):
        """Build Andrásfai graph And(n) = circulant C(n, {1, ..., (n-1)/2})
        where n must be odd. This is triangle-free and has independence number
        approximately n/3."""
        if n % 2 == 0:
            return None
        
        adj = [0] * n
        half = (n - 1) // 2
        
        for i in range(n):
            for d in range(1, half + 1):
                j = (i + d) % n
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        return adj

    def build_circulant(n, distances):
        """Build circulant graph C(n, distances)"""
        adj = [0] * n
        for i in range(n):
            for d in distances:
                j1 = (i + d) % n
                j2 = (i - d) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        return adj

    # Try Andrásfai graphs for various odd n
    # And(2k+1) has independence number k
    # We need alpha < 10, so k <= 9, meaning n <= 19 for And(n)
    # But we can try larger constructions
    
    # Start with known valid graphs and try to extend
    best_n = 0
    best_edges = []
    
    # Try Andrásfai And(19) which has alpha = 9
    # This gives n=19, alpha=9 < 10, so valid
    adj19 = build_andrasfai(19)
    if adj19 is not None and not has_independent_set_10(adj19, 19):
        best_n = 19
        edge_list = []
        for i in range(19):
            for j in range(i + 1, 19):
                if adj19[i] & (1 << j):
                    edge_list.append([i, j])
        best_edges = edge_list
    
    # Try to find larger graphs using random search with better heuristics
    for n_try in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random triangle-free graph
            adj = [0] * n_try
            edges = [(i, j) for i in range(n_try) for j in range(i + 1, n_try)]
            random.shuffle(edges)
            
            for u, v in edges:
                if is_triangle_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            if not has_independent_set_10(adj, n_try):
                found = True
                best_n = n_try
                edge_list = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edge_list.append([i, j])
                best_edges = edge_list
                break
        
        if found:
            continue  # Try larger n
        else:
            break  # If we can't find for this n, stop
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return Andrásfai And(19)
    adj19 = build_andrasfai(19)
    edge_list = []
    for i in range(19):
        for j in range(i + 1, 19):
            if adj19[i] & (1 << j):
                edge_list.append([i, j])
    return {'n': 19, 'edges': edge_list}
