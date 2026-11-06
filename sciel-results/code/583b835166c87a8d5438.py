import random
import time

def construct():
    start_time = time.time()
    time_limit = 115
    
    def try_construct(n):
        if time.time() - start_time > time_limit:
            return None
            
        def find_triangle(adj):
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        if common:
                            k_bit = common & -common
                            k = k_bit.bit_length() - 1
                            return (i, j, k)
            return None

        def find_independent_set_10(adj):
            all_mask = (1 << n) - 1
            nodes_visited = 0
            limit = 50000
            
            def backtrack(candidates, size, current_set):
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > limit:
                    return "TIMEOUT"
                if size == 10:
                    return current_set
                if not candidates:
                    return None
                if candidates.bit_count() < 10 - size:
                    return None
                
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                
                new_candidates = candidates & ~adj[v] & ~(1 << v)
                res = backtrack(new_candidates, size + 1, current_set + [v])
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res:
                    return res
                
                new_candidates = candidates & ~v_bit
                res = backtrack(new_candidates, size, current_set)
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res:
                    return res
                
                return None

            return backtrack(all_mask, 0, [])

        for attempt in range(5000):
            if time.time() - start_time > time_limit:
                return None
                
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            for step in range(500):
                tri = find_triangle(adj)
                if not tri:
                    break
                i, j, k = tri
                r = random.random()
                if r < 0.33:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                elif r < 0.66:
                    adj[i] &= ~(1 << k)
                    adj[k] &= ~(1 << i)
                else:
                    adj[j] &= ~(1 << k)
                    adj[k] &= ~(1 << j)
                    
            if find_triangle(adj):
                continue
                
            # Greedy check
            candidates = (1 << n) - 1
            greedy_count = 0
            while candidates and greedy_count < 10:
                min_deg = n
                best_v = -1
                c = candidates
                while c:
                    v_bit = c & -c
                    v = v_bit.bit_length() - 1
                    deg = (adj[v] & candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    c &= c - 1
                if best_v == -1:
                    break
                greedy_count += 1
                candidates &= ~adj[best_v]
                candidates &= ~(1 << best_v)
                
            if greedy_count < 10:
                res = find_independent_set_10(adj)
                if res == "TIMEOUT":
                    continue
                if res is None:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    u, v = random.sample(res, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        return None

    # Try n=40
    res = try_construct(40)
    if res:
        return res
        
    # Try n=39
    res = try_construct(39)
    if res:
        return res
        
    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
