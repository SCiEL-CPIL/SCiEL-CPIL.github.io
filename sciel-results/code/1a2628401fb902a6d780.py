import time
import random

def construct():
    deadline = time.time() + 140
    
    def edge_idx(i, j, n):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_triangles(colors, n):
        tri_list = []
        adj = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        for c in range(4):
            for i in range(n):
                for j in adj[c][i]:
                    if j <= i: continue
                    common = adj[c][i].intersection(adj[c][j])
                    for k in common:
                        if k > j:
                            tri_list.append((i, j, k, c))
        return tri_list

    def try_solve(n):
        if n < 1: return None
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        iterations = 0
        max_iterations = 20000
        while time.time() < deadline - 2 and iterations < max_iterations:
            iterations += 1
            tris = count_triangles(colors, n)
            if not tris:
                return colors
            
            i, j, k, c = random.choice(tris)
            edges_to_check = [(i, j), (i, k), (j, k)]
            random.shuffle(edges_to_check)
            
            # Greedy Min-Conflicts Step
            best_idx = -1
            best_new_c = -1
            best_count = float('inf')
            
            for u, v in edges_to_check:
                idx_uv = edge_idx(u, v, n)
                old_c = colors[idx_uv]
                
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    colors[idx_uv] = new_c
                    # Quick check: count triangles only? 
                    # To save time, we can just count triangles involving u or v
                    # But for simplicity and robustness, we do full count every few steps?
                    # No, full count is O(N^3). 12 calls * 0.01s = 0.12s.
                    # 100s / 0.12s = 800 steps.
                    # This is slow.
                    
                    # Let's use a cheaper metric: 
                    # Number of triangles containing edge (u,v)
                    # If we change (u,v) to new_c, we break triangles of old_c 
                    # and create triangles of new_c.
                    # Triangles of old_c containing (u,v): 
                    #   w such that (u,w) and (v,w) are old_c
                    # Triangles of new_c containing (u,v):
                    #   w such that (u,w) and (v,w) are new_c
                    
                    # This is O(N) per candidate. 12 * 50 = 600 ops. Very fast.
                    
                    break_count = 0
                    create_count = 0
                    for w in range(n):
                        if w == u or w == v: continue
                        idx_uw = edge_idx(u, w, n)
                        idx_vw = edge_idx(v, w, n)
                        
                        if colors[idx_uw] == old_c and colors[idx_vw] == old_c:
                            break_count += 1
                        if colors[idx_uw] == new_c and colors[idx_vw] == new_c:
                            create_count += 1
                            
                    net_change = create_count - break_count
                    if net_change < best_count:
                        best_count = net_change
                        best_idx = idx_uv
                        best_new_c = new_c
                        
                colors[idx_uv] = old_c
            
            if best_idx != -1:
                colors[best_idx] = best_new_c
            else:
                # Fallback to random
                u, v = random.choice(edges_to_check)
                idx_uv = edge_idx(u, v, n)
                old_c = colors[idx_uv]
                new_c = random.randint(0, 3)
                if new_c == old_c: new_c = (new_c + 1) % 4
                colors[idx_uv] = new_c
                
        return None

    def greedy_construct(n):
        if n < 1: return None
        m = n * (n - 1) // 2
        colors = [0] * m
        # We need to build the graph incrementally? 
        # The flat array makes incremental building hard because indices shift.
        # We can use a 2D array for greedy, then flatten.
        grid = [[-1]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                valid = True
                chosen_c = -1
                for c in range(4):
                    ok = True
                    for k in range(n):
                        if k == i or k == j: continue
                        # Only check if both (i,k) and (j,k) are already set
                        # This requires ordering. 
                        # If we fill row by row, when setting (i,j), 
                        # (i,k) for k<j is set. (j,k) for k<j is set.
                        # (i,k) for k>j is not set.
                        # So we can only check k < i? No.
                        # We can check k such that (i,k) and (j,k) are set.
                        # If we iterate i=0..n-1, j=i+1..n-1:
                        # When setting (i,j), 
                        # (i,k) is set for k < j.
                        # (j,k) is set for k < j.
                        # So we can check all k < j.
                        if k < j:
                            if grid[i][k] == c and grid[j][k] == c:
                                ok = False
                                break
                    if ok:
                        chosen_c = c
                        break
                if chosen_c == -1:
                    return None
                grid[i][j] = chosen_c
                grid[j][i] = chosen_c
        flat = []
        for i in range(n):
            for j in range(i+1, n):
                flat.append(grid[i][j])
        return flat

    # Try local search for 50
    res = try_solve(50)
    if res and _verify(50, res):
        return {'n': 50, 'colors': res}
        
    # Try local search for 49
    res = try_solve(49)
    if res and _verify(49, res):
        return {'n': 49, 'colors': res}
        
    # Try greedy for smaller n
    for target_n in [48, 40, 30, 20, 16]:
        if time.time() > deadline - 5: break
        res = greedy_construct(target_n)
        if res and _verify(target_n, res):
            return {'n': target_n, 'colors': res}
            
    return {'n': 1, 'colors': []}

def _verify(n, colors):
    if n == 1: return True
    if len(colors) != n * (n - 1) // 2: return False
    adj = [[False] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 3: return False
            adj[c][i][j] = adj[c][j][i] = True
            idx += 1
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[c][i][j]:
                    continue
                for k in range(j + 1, n):
                    if adj[c][i][k] and adj[c][j][k]:
                        return False
    return True
