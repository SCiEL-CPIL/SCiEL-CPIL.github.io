import time
import sys
sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def attempt(n, time_budget):
        local_deadline = time.time() + time_budget
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        # For each color, track for each pair (u,v) the number of common neighbors
        # Actually, let's use a different approach for C6 detection.
        # We need to check if adding (u,v) creates a C6.
        # This happens if there is a path of length 5 between u and v in color c.
        # A path of length 5 is u-x1-x2-x3-x4-v.
        
        # To speed up, we can precompute for each color c, and for each pair (u,v),
        # the set of intermediate nodes that can form a path of length 5.
        # But this is expensive.
        
        # Alternative: Use BFS/DFS with pruning.
        # Since n <= 17, the graph is small.
        
        def has_path_len_5(c, u, v):
            # Check for simple path of length 5 from u to v in color c
            # u - x1 - x2 - x3 - x4 - v
            # We can do DFS from u with depth 5, trying to reach v.
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    return (adj[c][cur] >> v) & 1
                if depth == 2:
                    # Path cur - a - v. Need common neighbor.
                    return (adj[c][cur] & adj[c][v]) != 0
                if depth == 3:
                    # Path cur - a - b - v.
                    # For each neighbor a of cur, check if a has common neighbor with v (excluding cur, v).
                    # Actually, let's just do DFS for depth 3 and 4.
                    pass
                
                mask = adj[c][cur] & ~visited_mask
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if dfs(nxt, depth - 1, visited_mask | lsb):
                        return True
                    mask -= lsb
                return False

            return dfs(u, 5, 1 << u)

        def is_valid(c, u, v):
            return not has_path_len_5(c, u, v)

        def solve(idx):
            if time.time() > local_deadline:
                return False
            if idx == num_edges:
                return True
                
            u, v = edges[idx]
            for c in range(4):
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if solve(idx + 1):
                        return True
                        
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            return False

        if solve(0):
            return colors
        return None

    # Try n=17 first
    res = attempt(17, 110)
    if res:
        return {'n': 17, 'colors': res}

    # Fallback to n=16
    remaining = hard_deadline - time.time()
    if remaining > 10:
        res = attempt(16, min(10, remaining - 2))
        if res:
            return {'n': 16, 'colors': res}

    # Fallback to n=15
    remaining = hard_deadline - time.time()
    if remaining > 5:
        res = attempt(15, min(5, remaining - 2))
        if res:
            return {'n': 15, 'colors': res}

    # Absolute fallback: small valid coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((j - i) % 4)
    return {'n': n, 'colors': colors}
