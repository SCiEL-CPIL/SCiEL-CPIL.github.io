import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def is_c4_free_add(neighbors, u, v):
        if (neighbors[u] >> v) & 1:
            return False
        if neighbors[u] & neighbors[v]:
            return False
        return True

    def add_edge(neighbors, u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    def greedy_alpha(neighbors, n):
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            min_deg = n
            min_v = -1
            r = remaining
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                deg = (neighbors[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            count += 1
            remaining &= ~(neighbors[min_v] | (1 << min_v))
        return count

    def exact_alpha(neighbors, n, time_limit=1.0):
        start = time.time()
        best = 0
        # Use a simple DFS with pruning
        def dfs(remaining, current_size):
            nonlocal best
            if time.time() - start > time_limit:
                return -1 # Timeout
            if current_size > best:
                best = current_size
            if not remaining:
                return 0
            if current_size + remaining.bit_count() <= best:
                return 0
            
            # Pick vertex with max degree
            v = -1
            max_deg = -1
            r = remaining
            while r:
                cv = (r & -r).bit_length() - 1
                r &= r - 1
                deg = (neighbors[cv] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = cv
            
            # Include v
            res1 = dfs(remaining & ~(neighbors[v] | (1 << v)), current_size + 1)
            if res1 == -1: return -1
            # Exclude v
            res2 = dfs(remaining & ~(1 << v), current_size)
            if res2 == -1: return -1
            return 0
        
        dfs((1 << n) - 1, 0)
        if time.time() - start > time_limit:
            return -1
        return best

    def try_construct(n, seed):
        random.seed(seed)
        neighbors = [0] * n
        edges = []
        
        # Generate random order of edges
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if is_c4_free_add(neighbors, u, v):
                add_edge(neighbors, u, v)
                edges.append([u, v])
                
        g_alpha = greedy_alpha(neighbors, n)
        if g_alpha <= 10:
            e_alpha = exact_alpha(neighbors, n, time_limit=2.0)
            if e_alpha != -1 and e_alpha <= 10:
                return {'n': n, 'edges': edges}
        
        return None

    # Try n from 39 down to 30
    # We prioritize higher n
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
            
        seed = 0
        found = False
        while time.time() < deadline - 5.0: # Leave 5s for fallback
            res = try_construct(n, seed)
            if res:
                return res
            seed += 1
            # If we've tried 500 seeds for this n, move to next n
            if seed > 500:
                break

    # Fallback: Return seed result
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
