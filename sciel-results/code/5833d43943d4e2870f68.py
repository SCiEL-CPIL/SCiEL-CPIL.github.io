import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def check_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u+1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    def has_independent_set_11(adj_mask, n, time_limit=2.0):
        # Returns True if there is an independent set of size 11
        # Returns False if proven no such set exists
        # Returns None if timeout (inconclusive)
        
        target = 11
        start_check = time.time()
        
        # Greedy heuristic first: try to find an independent set of size 11
        # If found, return True immediately.
        # Greedy: pick vertex with min degree in current candidate set, add to set, remove neighbors.
        candidates = (1 << n) - 1
        current_set_size = 0
        greedy_candidates = candidates
        while greedy_candidates:
            if time.time() - start_check > time_limit:
                return None
            # Find min degree vertex in greedy_candidates
            min_deg = n
            chosen = -1
            c = greedy_candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & greedy_candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            if chosen == -1:
                break
            current_set_size += 1
            if current_set_size >= target:
                return True
            greedy_candidates &= ~(adj_mask[chosen] | (1 << chosen))
        
        if current_set_size >= target:
            return True
            
        # If greedy didn't find 11, we must prove no 11 exists.
        # Use DFS with branch and bound.
        best = [0]
        
        def dfs(candidates, size):
            if time.time() - start_check > time_limit:
                return None # Timeout
            
            if size >= target:
                return True
            if candidates == 0:
                return False
            
            # Upper bound
            if size + candidates.bit_count() < target:
                return False
            
            # Heuristic: Pick vertex with max degree in candidates
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return False

            # Branch 1: Include chosen
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            res = dfs(new_candidates, size + 1)
            if res is True:
                return True
            if res is None:
                return None
            
            # Branch 2: Exclude chosen
            new_candidates = candidates & ~(1 << chosen)
            res = dfs(new_candidates, size)
            if res is True:
                return True
            if res is None:
                return None
            
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)

    def generate_greedy_c4_free(n, seed):
        random.seed(seed)
        adj_mask = [0] * n
        graph_edges = []
        
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            nu = adj_mask[u]
            nv = adj_mask[v]
            
            # Check if adding (u,v) creates a C4
            # Condition: No edge between N(u) and N(v)
            
            if nv.bit_count() < nu.bit_count():
                c = nv
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nu:
                        safe = False
                        break
            else:
                c = nu
                safe = True
                while c:
                    lsb = c & (-c)
                    x = lsb.bit_length() - 1
                    c ^= lsb
                    if adj_mask[x] & nv:
                        safe = False
                        break
            
            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                graph_edges.append([u, v])
                
        return adj_mask, graph_edges

    # Try n = 40, 39, ...
    # We want to maximize n.
    # Best known is 39? No, the prompt says "Best known construction has 39 vertices" but 
    # the score 1.0333 implies n=31 was found by the seed (31/30).
    # Wait, the prompt says "Maximize n. A valid graph on n vertices certifies R(C4,K11) > n."
    # If the seed found 30, and score is 1.0333, that's 31/30.
    # So we are looking for n > 31.
    
    best_n = 30
    best_edges = []
    
    # Fallback edges for n=30 (10 triangles)
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    for n in range(43, 30, -1):
        # If we find a valid graph for n, we return it immediately as it's better.
        # But we should try to find the largest n possible.
        # Since we iterate downwards, the first valid n is the max.
        
        # Number of seeds to try for this n
        # If n is large, it's harder to satisfy alpha < 11.
        # Let's try a decent number of seeds.
        seeds_to_try = 100 if n <= 35 else 20
        
        for seed in range(seeds_to_try):
            if time.time() > deadline - 5:
                break
            
            adj_mask, graph_edges = generate_greedy_c4_free(n, seed)
            
            # Verify C4-free
            if not check_c4_free(adj_mask, n):
                continue
                
            # Check if alpha < 11
            # We use a time limit for the exact check to avoid hanging.
            # If it returns True, we have an independent set of 11 -> Invalid.
            # If it returns False, we proved no independent set of 11 -> Valid.
            # If it returns None, timeout -> Skip this seed.
            
            res = has_independent_set_11(adj_mask, n, time_limit=1.0)
            
            if res is False:
                # Valid graph found
                return {'n': n, 'edges': graph_edges}
            
        # If we didn't find a valid graph for this n, try smaller n.
        # But we must ensure we return the best found so far if we run out of time.
        # The loop structure returns immediately on success, so if we finish the loop for n,
        # we just move to n-1.
        
    return {'n': best_n, 'edges': best_edges}
