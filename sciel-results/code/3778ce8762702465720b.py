import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    def get_edges(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def solve(n):
        edges = get_edges(n)
        m = len(edges)
        # Map edge to index
        edge_index = {}
        for idx, (u, v) in enumerate(edges):
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
        # Initialize random colors
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Precompute common neighbors counts
        # common[u][v][c] = number of w such that (u,w) color c and (v,w) color c
        # We can store this as a list of lists of lists
        # Size n x n x 4
        common = [[[0 for _ in range(4)] for _ in range(n)] for _ in range(n)]
        
        def calc_common():
            # Reset
            for i in range(n):
                for j in range(n):
                    for c in range(4):
                        common[i][j][c] = 0
            # Fill
            for u in range(n):
                for v in range(u + 1, n):
                    idx_uv = edge_index[(u, v)]
                    # We need to count for each w
                    # Actually, it's easier to iterate w and update
            # Better: Iterate all triangles (u, v, w)
            # For each triangle, check colors of 3 edges
            # If all 3 are color c, increment common[u][v][c], common[u][w][c], common[v][w][c]
            # Wait, common[u][v][c] counts w. So if (u,v)=c, (u,w)=c, (v,w)=c, then w contributes to common[u][v][c].
            # So for each triangle (u,v,w):
            #   c_uv = colors[edge_index[(u,v)]]
            #   c_uw = colors[edge_index[(u,w)]]
            #   c_vw = colors[edge_index[(v,w)]]
            #   If c_uv == c_uw == c_vw == c:
            #       common[u][v][c] += 1
            #       common[v][u][c] += 1
            #       common[u][w][c] += 1
            #       common[w][u][c] += 1
            #       common[v][w][c] += 1
            #       common[w][v][c] += 1
            # This is O(n^3) which is 125k for n=50. Fine.
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        c_uv = colors[edge_index[(u, v)]]
                        c_uw = colors[edge_index[(u, w)]]
                        c_vw = colors[edge_index[(v, w)]]
                        if c_uv == c_uw == c_vw:
                            c = c_uv
                            common[u][v][c] += 1
                            common[v][u][c] += 1
                            common[u][w][c] += 1
                            common[w][u][c] += 1
                            common[v][w][c] += 1
                            common[w][v][c] += 1

        calc_common()
        
        # Calculate initial E (total monochromatic triangles)
        # E = sum over all triangles of 1 if mono else 0
        # Or E = sum_c (number of mono triangles of color c)
        # Note: sum_c common[u][v][c] for a fixed (u,v) is the number of w completing a mono triangle with uv.
        # So E = 0.5 * sum_{u<v} sum_c common[u][v][c] ? No.
        # Each mono triangle (u,v,w) is counted 3 times in the sum over edges?
        # Let's just count directly or use the common array.
        # Sum over all pairs (u,v) and color c of common[u][v][c] counts each triangle 3 times (once for each edge).
        # So E = (1/3) * sum_{u<v} sum_c common[u][v][c]
        E = 0
        for u in range(n):
            for v in range(u + 1, n):
                for c in range(4):
                    E += common[u][v][c]
        E //= 3
        
        if E == 0:
            return colors
            
        # Local Search
        best_colors = colors[:]
        best_E = E
        
        # Simulated Annealing or Steepest Descent
        # Let's use Random Walk with Backtracking to best if stuck
        # Or just Simple Local Search with Random Restarts
        
        steps = 0
        while time.time() - start_time < time_limit:
            # Pick random edge
            idx = random.randint(0, m - 1)
            u, v = edges[idx]
            c_old = colors[idx]
            
            # Find best new color
            best_delta = 0
            best_c = c_old
            for c_new in range(4):
                if c_new == c_old:
                    continue
                delta = common[u][v][c_new] - common[u][v][c_old]
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            if best_c == c_old:
                # No improvement, try random move or restart
                # Let's do a random move to escape local minima
                c_new = random.randint(0, 3)
                if c_new == c_old:
                    continue
                delta = common[u][v][c_new] - common[u][v][c_old]
            else:
                c_new = best_c
                delta = best_delta
                
            # Apply move
            colors[idx] = c_new
            
            # Update E
            E += delta
            
            # Update common
            # For each w != u, v:
            # If (u,w) is c_old, common[u][w][c_old]--
            # If (u,w) is c_new, common[u][w][c_new]++
            # Same for (v,w)
            for w in range(n):
                if w == u or w == v:
                    continue
                # Edge (u, w)
                idx_uw = edge_index[(u, w)]
                c_uw = colors[idx_uw]
                if c_uw == c_old:
                    common[u][w][c_old] -= 1
                    common[w][u][c_old] -= 1
                elif c_uw == c_new:
                    common[u][w][c_new] += 1
                    common[w][u][c_new] += 1
                    
                # Edge (v, w)
                idx_vw = edge_index[(v, w)]
                c_vw = colors[idx_vw]
                if c_vw == c_old:
                    common[v][w][c_old] -= 1
                    common[w][v][c_old] -= 1
                elif c_vw == c_new:
                    common[v][w][c_new] += 1
                    common[w][v][c_new] += 1
            
            # Update edge (u,v) in common?
            # common[u][v][c] does not depend on color of (u,v).
            # But wait, my definition of common[u][v][c] was:
            # count of w such that (u,w)=c and (v,w)=c.
            # This does NOT involve (u,v).
            # So common[u][v][c] is unchanged by flipping (u,v).
            
            if E < best_E:
                best_E = E
                best_colors = colors[:]
                if E == 0:
                    return best_colors
                    
            steps += 1
            # If stuck for too long, restart?
            # For now, just keep going. Random moves help escape.
            
        return best_colors if best_E == 0 else None

    # Try n from 50 down to 33
    for n in range(50, 32, -1):
        if time.time() - start_time > time_limit - 10:
            break
        res = solve(n)
        if res is not None:
            # Convert to list format required
            final_colors = res
            return {'n': n, 'colors': final_colors}
            
    # Fallback to seed solution n=32
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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
