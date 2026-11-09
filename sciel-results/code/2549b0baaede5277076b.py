import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def is_safe_check(adj, c, u, v):
        visited = 1 << u
        frontier = 1 << u
        target = 1 << v
        for step in range(5):
            next_frontier = 0
            f = frontier
            while f:
                lsb = f & -f
                node = lsb.bit_length() - 1
                next_frontier |= adj[c][node]
                f ^= lsb
            next_frontier &= ~visited
            if step == 4:
                if next_frontier & target:
                    return False
            visited |= next_frontier
            frontier = next_frontier
            if frontier == 0:
                return True
        return True

    def solve_backtrack(n, budget):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * len(edges)
        
        def backtrack(idx):
            if time.time() - start_time > (start_time + budget):
                return False
            if idx == len(edges):
                return True
            u, v = edges[idx]
            
            if idx == 0:
                candidates = [0]
            elif idx == 1:
                candidates = [0, 1]
            elif idx == 2:
                candidates = [0, 1, 2]
            else:
                candidates = [0, 1, 2, 3]
                
            random.shuffle(candidates)
            
            for c in candidates:
                if is_safe_check(adj, c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            return False

        if backtrack(0):
            return {'n': n, 'colors': colors}
        return None

    def solve_greedy(n, budget):
        start = time.time()
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for attempt in range(50):
            if time.time() - start > budget:
                break
            # Shuffle edge order for greedy
            order = list(range(len(edges)))
            random.shuffle(order)
            
            adj = [[0] * n for _ in range(4)]
            colors = [-1] * len(edges)
            
            success = True
            for idx in order:
                u, v = edges[idx]
                found = False
                for c in range(4):
                    if is_safe_check(adj, c, u, v):
                        colors[idx] = c
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                return {'n': n, 'colors': colors}
        
        return None

    # Try n=17
    res = solve_backtrack(17, 100.0)
    if res:
        return res

    # Try n=16
    res = solve_backtrack(16, 10.0)
    if res:
        return res

    # Try n=15 with greedy
    res = solve_greedy(15, 5.0)
    if res:
        return res

    # Try n=14 with greedy
    res = solve_greedy(14, 5.0)
    if res:
        return res

    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
