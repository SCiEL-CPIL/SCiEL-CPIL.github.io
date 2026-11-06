import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (150s limit)
    
    def check_c4_free(adj, n):
        """Check if graph is C4-free using bitmask representation."""
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        for u in range(n):
            for v in range(u+1, n):
                common = adj_mask[u] & adj_mask[v]
                if bin(common).count('1') >= 2:
                    return False
        return True
    
    # Use incidence graph of PG(2,2): 14 vertices, C4-free, bipartite
    # 7 points + 7 lines, edges connect point-line incidence
    # PG(2,2) points: (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)
    # Lines in PG(2,2) correspond to 2D subspaces
    pg22_points = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    # Lines: each line has 3 points. There are 7 lines in PG(2,2)
    pg22_lines = [
        [(1,0,0),(0,1,0),(1,1,0)],
        [(1,0,0),(0,0,1),(1,0,1)],
        [(1,0,0),(0,1,1),(1,1,1)],
        [(0,1,0),(0,0,1),(0,1,1)],
        [(0,1,0),(1,0,1),(1,1,1)],
        [(0,0,1),(1,1,0),(1,0,1)],
        [(1,1,0),(0,1,1),(1,1,1)]
    ]
    
    n_inc = 14
    adj_inc = [[False]*n_inc for _ in range(n_inc)]
    edges_inc = []
    for li, line in enumerate(pg22_lines):
        for pt in line:
            pi = pg22_points.index(pt)
            adj_inc[pi][li + 7] = True
            adj_inc[li + 7][pi] = True
            edges_inc.append([pi, li + 7])
    
    # Verify C4-free
    if not check_c4_free(adj_inc, n_inc):
        # Should never happen, but fallback to empty graph
        return {'n': 1, 'edges': []}
    
    # This graph has alpha = 7 (each part of bipartition is independent)
    # We need alpha < 11. Single copy: alpha=7 < 11, n=14
    # Two copies: alpha=14 >= 11, invalid
    # Three copies: alpha=21 >= 11, invalid
    # So single copy gives n=14, which is valid but small
    
    # Better: use a C4-free graph with smaller independence number
    # Try polarity graph of PG(2,4) but verify it
    mult_table = [
        [0, 1, 2, 3],
        [1, 1, 2, 3],
        [2, 2, 3, 1],
        [3, 3, 1, 2]
    ]
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        return mult_table[a][b]
    
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                vec = (x, y, z)
                is_new = True
                for p in points:
                    for k in [1, 2, 3]:
                        if (gf4_mul(k, p[0]) == vec[0] and 
                            gf4_mul(k, p[1]) == vec[1] and 
                            gf4_mul(k, p[2]) == vec[2]):
                            is_new = False
                            break
                    if not is_new:
                        break
                if is_new:
                    points.append(vec)
    
    n4 = len(points)
    
    adj4 = [[False]*n4 for _ in range(n4)]
    edges4 = []
    for i in range(n4):
        for j in range(i+1, n4):
            u = points[i]
            v = points[j]
            dot = gf4_add(gf4_add(gf4_mul(u[0], v[2]), gf4_mul(u[1], v[1])), gf4_mul(u[2], v[0]))
            if dot == 0:
                adj4[i][j] = True
                adj4[j][i] = True
                edges4.append([i, j])
    
    # Verify C4-free
    if check_c4_free(adj4, n4):
        # Compute independence number
        def max_independent_set_fast(adj, n):
            adj_mask = [0] * n
            for i in range(n):
                for j in range(n):
                    if adj[i][j]:
                        adj_mask[i] |= (1 << j)
            best = [0]
            
            def dfs(candidates, size):
                if size > best[0]:
                    best[0] = size
                if candidates == 0:
                    return
                if size + bin(candidates).count('1') <= best[0]:
                    return
                max_deg = -1
                chosen = -1
                c = candidates
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = bin(adj_mask[v] & candidates).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        chosen = v
                if chosen == -1:
                    return
                new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
                dfs(new_candidates, size + 1)
                new_candidates = candidates & ~(1 << chosen)
                dfs(new_candidates, size)
            
            all_candidates = (1 << n) - 1
            dfs(all_candidates, 0)
            return best[0]
        
        alpha4 = max_independent_set_fast(adj4, n4)
        
        # If alpha4 < 11, we can use this graph directly
        if alpha4 < 11:
            return {'n': n4, 'edges': edges4}
        else:
            # alpha4 >= 11, can't use directly. Fall back to incidence graph
            return {'n': n_inc, 'edges': edges_inc}
    else:
        # PG(2,4) polarity graph is not C4-free, use incidence graph
        return {'n': n_inc, 'edges': edges_inc}
