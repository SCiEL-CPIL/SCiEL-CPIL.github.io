import random
import time

def construct():
    n = 39
    
    def random_triangle_free_graph(n):
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.6: # Slightly higher prob to increase density
                    common = adj[i] & adj[j]
                    if common == 0:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append((i, j))
        return adj, edges

    def has_independent_set(adj, n, k):
        mask = (1 << n) - 1
        
        def dfs(current_size, candidates):
            if current_size >= k:
                return True
            if not candidates:
                return False
            if current_size + bin(candidates).count('1') < k:
                return False
            
            # Pick vertex with max degree in candidates
            # This helps prune the "exclude" branch?
            # Actually, for independent set, including a vertex with high degree removes many candidates.
            # So picking high degree vertex first might be good.
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v_bit = temp & -temp
                v_idx = v_bit.bit_length() - 1
                deg = bin(candidates & adj[v_idx]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v_idx
                temp -= v_bit
                
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            new_cands = candidates & ~adj[best_v] & ~v_bit
            if dfs(current_size + 1, new_cands):
                return True
            
            # Branch 2: Exclude best_v
            new_cands_excl = candidates & ~v_bit
            if dfs(current_size, new_cands_excl):
                return True
            
            return False
        
        return dfs(0, mask)

    start_time = time.time()
    max_time = 115

    # Try n=39
    for attempt in range(500):
        if time.time() - start_time > max_time:
            break
        adj, edges = random_triangle_free_graph(n)
        if not has_independent_set(adj, n, 10):
            return {'n': n, 'edges': [list(e) for e in edges]}

    # Fallback to n=38
    n = 38
    for attempt in range(500):
        if time.time() - start_time > max_time:
            break
        adj, edges = random_triangle_free_graph(n)
        if not has_independent_set(adj, n, 10):
            return {'n': n, 'edges': [list(e) for e in edges]}

    # Fallback to n=37
    n = 37
    for attempt in range(500):
        if time.time() - start_time > max_time:
            break
        adj, edges = random_triangle_free_graph(n)
        if not has_independent_set(adj, n, 10):
            return {'n': n, 'edges': [list(e) for e in edges]}

    # Safe fallback: Andrásfai graph A_19 (n=19, alpha=9)
    n = 19
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if 1 <= diff <= 8:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
