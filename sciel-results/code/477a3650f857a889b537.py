import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Seed solution for n=32
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
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    best_n = 32
    best_colors = seed_colors

    def try_find_coloring(n, max_iters=5000):
        """
        Try to find a valid 4-coloring of K_n using local search.
        Returns list of colors or None.
        """
        # Initialize random coloring
        # Use NumPy for speed if possible, but pure Python might be safer for small n
        # n=50, edges=1225. 
        # We will use a list of colors for edges.
        # Indexing: edge (i, j) where i < j.
        # Index = i * n - i*(i+1)//2 + (j - i - 1) ? 
        # Simpler: store in a 2D array or a flat list with a mapping.
        # Let's use a 2D array color[i][j] for simplicity, n<=50 is small.
        
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
        
        # Precompute N_c(i, j) for all edges and all colors
        # N[c][i][j] = count of k such that colors[i][k]==c and colors[j][k]==c
        # This is a 4 * n * n array.
        N = np.zeros((4, n, n), dtype=np.int32)
        
        def update_N():
            # Full recalculation
            N.fill(0)
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        if k == i or k == j: continue
                        c = colors[i][k]
                        if c == colors[j][k]:
                            N[c][i][j] += 1
                            N[c][j][i] += 1
                            
        update_N()
        
        def calc_T():
            t = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[i][j]
                    t += N[c][i][j]
            return t // 3

        T = calc_T()
        
        if T == 0:
            return [colors[i][j] for i in range(n) for j in range(i + 1, n)]
            
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            best_delta = 0
            best_u = -1
            best_v = -1
            best_new_c = -1
            
            # Sample a subset of edges to flip to save time?
            # Or iterate all? n=50, edges=1225. 1225 * 3 (other colors) = 3675 evaluations.
            # Each eval is O(n) = 50. Total 180k ops per iteration.
            # 5000 iters = 900M ops. Too slow in Python.
            # We need to speed up.
            
            # Optimization: Only consider edges that are part of many triangles?
            # Or random sampling of edges.
            
            # Let's try random sampling of 100 edges per iteration.
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            sampled = random.sample(edges, min(100, len(edges)))
            
            for u, v in sampled:
                c_old = colors[u][v]
                for c_new in range(4):
                    if c_new == c_old: continue
                    
                    # Calculate delta T
                    delta = 0
                    # Change in sum for edge (u,v):
                    # Old term: N[c_old][u][v]
                    # New term: N[c_new][u][v]
                    # But N[c][u][v] does not depend on colors[u][v].
                    # So the term for (u,v) in the sum changes from N[c_old][u][v] to N[c_new][u][v].
                    delta += N[c_new][u][v] - N[c_old][u][v]
                    
                    # Change in terms for (u, x) and (v, x)
                    for x in range(n):
                        if x == u or x == v: continue
                        
                        # Update N for (u, x)
                        c_ux = colors[u][x]
                        # The count N[c_ux][u][x] changes by:
                        # -1 if colors[v][x] == c_old (because (u,v) was c_old, so if (v,x) was c_old, it contributed to N[c_old][u][x]? No.)
                        # Let's re-derive.
                        # N[c][u][x] = count of y s.t. colors[u][y]==c and colors[x][y]==c.
                        # The term for y=v is: colors[u][v]==c and colors[x][v]==c.
                        # If we change colors[u][v] from c_old to c_new:
                        # If c_ux == c_old:
                        #   Old contribution: 1 if colors[x][v] == c_old.
                        #   New contribution: 0 (since colors[u][v] is no longer c_old).
                        #   So delta_N_ux = -1 if colors[x][v] == c_old else 0.
                        # If c_ux == c_new:
                        #   Old contribution: 0.
                        #   New contribution: 1 if colors[x][v] == c_new.
                        #   So delta_N_ux = 1 if colors[x][v] == c_new else 0.
                        
                        d_ux = 0
                        if c_ux == c_old:
                            if colors[x][v] == c_old: d_ux -= 1
                        if c_ux == c_new:
                            if colors[x][v] == c_new: d_ux += 1
                            
                        delta += d_ux
                        
                        # Similarly for (v, x)
                        c_vx = colors[v][x]
                        d_vx = 0
                        if c_vx == c_old:
                            if colors[x][u] == c_old: d_vx -= 1
                        if c_vx == c_new:
                            if colors[x][u] == c_new: d_vx += 1
                            
                        delta += d_vx
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_u = u
                        best_v = v
                        best_new_c = c_new
                        
            if best_delta < 0:
                # Apply flip
                u, v, c_new = best_u, best_v, best_new_c
                c_old = colors[u][v]
                colors[u][v] = c_new
                colors[v][u] = c_new
                
                # Update N
                for x in range(n):
                    if x == u or x == v: continue
                    c_ux = colors[u][x]
                    if c_ux == c_old:
                        if colors[x][v] == c_old:
                            N[c_old][u][x] -= 1
                            N[c_old][x][u] -= 1
                    if c_ux == c_new:
                        if colors[x][v] == c_new:
                            N[c_new][u][x] += 1
                            N[c_new][x][u] += 1
                            
                    c_vx = colors[v][x]
                    if c_vx == c_old:
                        if colors[x][u] == c_old:
                            N[c_old][v][x] -= 1
                            N[c_old][x][v] -= 1
                    if c_vx == c_new:
                        if colors[x][u] == c_new:
                            N[c_new][v][x] += 1
                            N[c_new][x][v] += 1
                            
                T += best_delta // 3 # Approx? 
                # Actually T = Sum / 3. Delta Sum = best_delta * 3?
                # No, delta was calculated as the change in the SUM.
                # So T changes by delta / 3?
                # Wait, my delta calculation summed the changes in N values.
                # The sum S = sum N[c(e)][e].
                # Delta S = best_delta.
                # T = S / 3.
                # So T_new = (S + best_delta) / 3.
                # We should maintain S to be precise.
                pass # We will recalc T occasionally or just check if T becomes 0.
                
                # To avoid drift, recalc T every 100 steps or if we think it's close.
                if it % 50 == 0:
                    T = calc_T()
                    if T == 0:
                        return [colors[i][j] for i in range(n) for j in range(i + 1, n)]
            else:
                # Stuck, restart or perturb
                # Perturb: flip 5 random edges
                for _ in range(5):
                    u, v = random.sample(range(n), 2)
                    if u > v: u, v = v, u
                    colors[u][v] = random.randint(0, 3)
                    colors[v][u] = colors[u][v]
                update_N()
                T = calc_T()
                if T == 0:
                    return [colors[i][j] for i in range(n) for j in range(i + 1, n)]

        return None

    # Try to find coloring for n from 50 down to 33
    for n in range(50, 32, -1):
        if time.time() - start_time > time_limit:
            break
        # Try a few times
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            res = try_find_coloring(n)
            if res is not None:
                return {'n': n, 'colors': res}
                
    return {'n': best_n, 'colors': best_colors}
