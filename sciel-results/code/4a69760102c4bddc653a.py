import time
import sys

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def attempt(n, time_budget):
        local_deadline = time.time() + time_budget
        
        # Edges in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        colors = [-1] * num_edges
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        
        # Precompute edge index for fast lookup if needed, but we iterate in order.
        
        def has_path_len_5(c, u, v):
            # Check if there is a simple path of length 5 from u to v in color c
            # u - x1 - x2 - x3 - x4 - v
            # DFS with bitmask
            # visited_mask includes u initially
            
            def dfs(cur, depth, visited_mask):
                if depth == 0:
                    return cur == v
                if depth == 1:
                    return (adj[c][cur] >> v) & 1
                
                mask = adj[c][cur] & ~visited_mask
                # Optimization: if depth is large, check if v is reachable?
                # Just iterate
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if dfs(nxt, depth - 1, visited_mask | lsb):
                        return True
                    mask -= lsb
                return False

            return dfs(u, 5, 1 << u)

        def is_valid(c, u, v):
            # Adding (u,v) to color c is valid if no path of length 5 exists
            return not has_path_len_5(c, u, v)

        # Symmetry breaking
        # For n >= 5, fix colors of edges from 0 to 1, 2, 3, 4
        # This reduces the search space significantly.
        # We will handle these in the main loop or pre-assign them.
        # To keep the generic solver simple, we will just let the solver run,
        # but we can force the first few assignments.
        
        # Let's implement a recursive solver that takes an index
        # and uses symmetry breaking for the first few edges.
        
        def solve(idx, depth=0):
            if time.time() > local_deadline:
                return False
            
            if idx == num_edges:
                return True
                
            u, v = edges[idx]
            
            # Symmetry breaking logic
            # If u == 0 and v in [1, 2, 3, 4], restrict colors
            # We want color(0,1)=0, color(0,2)=1, color(0,3)=2, color(0,4)=3
            # This is a strong constraint.
            
            allowed_colors = [0, 1, 2, 3]
            if u == 0:
                if v == 1 and n > 1:
                    allowed_colors = [0]
                elif v == 2 and n > 2:
                    allowed_colors = [1]
                elif v == 3 and n > 3:
                    allowed_colors = [2]
                elif v == 4 and n > 4:
                    allowed_colors = [3]
            
            for c in allowed_colors:
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if solve(idx + 1, depth + 1):
                        return True
                        
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    
                    # If we broke symmetry, we don't need to try other colors for this edge
                    # because they are symmetric.
                    if u == 0 and v in [1, 2, 3, 4]:
                        break

            return False

        if solve(0):
            return colors
        return None

    # Try n=17
    res = attempt(17, 100)
    if res:
        return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 10:
        res = attempt(16, 10)
        if res:
            return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 5:
        res = attempt(15, 5)
        if res:
            return {'n': 15, 'colors': res}

    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
