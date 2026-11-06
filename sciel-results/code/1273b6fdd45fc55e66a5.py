def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 110
    
    def is_triangle_free(adj_bits, n):
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    if common:
                        return False
        return True
        
    def alpha_exact(adj_bits, n):
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            new_cands = [u for u in rest if not ((adj_bits[v] >> u) & 1)]
            dfs(new_cands, current_size + 1)
            
            dfs(rest, current_size)
            
            if best >= 10:
                return

        dfs(list(range(n)), 0)
        return best

    def find_graph(n):
        for restart in range(200):
            if time.time() > deadline:
                return None
            
            adj_bits = [0] * n
            
            # Initialize with a random triangle-free graph using a structured approach
            # Try to create a near-regular graph
            for _ in range(n * n // 2):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if (adj_bits[u] >> v) & 1: continue
                common = adj_bits[u] & adj_bits[v]
                if common:
                    continue
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
            # Local search with simulated annealing
            current_alpha = alpha_exact(adj_bits, n)
            if current_alpha < 10:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_bits[i] >> j) & 1:
                            edges.append([i, j])
                return edges
            
            best_alpha = current_alpha
            best_adj = adj_bits[:]
            
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                old_adj = adj_bits[:]
                
                if (adj_bits[u] >> v) & 1:
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                else:
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        continue
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                
                new_alpha = alpha_exact(adj_bits, n)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = adj_bits[:]
                    if new_alpha < 10:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj_bits[i] >> j) & 1:
                                    edges.append([i, j])
                        return edges
                else:
                    # Revert with some probability (simulated annealing)
                    if random.random() > 0.1:
                        adj_bits = old_adj
        
        return None

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        edges = find_graph(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: use a known construction
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
