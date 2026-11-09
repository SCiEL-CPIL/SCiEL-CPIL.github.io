import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate base 32-vertex coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            base_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    n_base = 32
    adj = [[0] * n_base for _ in range(4)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c][i] |= 1 << j
            adj[c][j] |= 1 << i
            idx += 1
            
    best_n = n_base
    best_colors = base_colors
    
    # Helper to try adding a vertex
    def try_add_vertex(n_cur, adj_in, colors_in):
        assignment = [-1] * n_cur
        assigned = [0, 0, 0, 0]
        
        def rec(i):
            if i == n_cur:
                return True
            for c in range(4):
                if (adj_in[c][i] & assigned[c]) == 0:
                    assignment[i] = c
                    assigned[c] |= (1 << i)
                    if rec(i + 1):
                        return True
                    assigned[c] ^= (1 << i)
                    assignment[i] = -1
            return False
            
        if rec(0):
            new_adj = [list(row) for row in adj_in]
            for c in range(4):
                new_adj[c].append(0)
            
            new_colors = list(colors_in)
            for i in range(n_cur):
                c = assignment[i]
                new_adj[c][i] |= (1 << n_cur)
                new_adj[c][n_cur] |= (1 << i)
                new_colors.append(c)
            return (True, new_adj, new_colors)
        else:
            return (False, None, None)

    current_n = n_base
    current_adj = adj
    current_colors = base_colors
    
    # Try to extend
    while current_n < 62:
        if time.time() > deadline:
            break
        
        success, new_adj, new_colors = try_add_vertex(current_n, current_adj, current_colors)
        if success:
            current_n += 1
            current_adj = new_adj
            current_colors = new_colors
            best_n = current_n
            best_colors = current_colors
        else:
            # If greedy fails, we might try a few random restarts of the last vertex assignment?
            # The backtracking in `try_add_vertex` is exhaustive for the new vertex given the fixed past.
            # So if it fails, it's impossible to add this vertex with the current past.
            # We would need to backtrack the past vertices.
            # For now, we stop.
            break
            
    return {'n': best_n, 'colors': best_colors}
