import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed program's result for fallback (n=32)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def seed_coloring():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = seed_coloring()
    best_n = 32

    def solve_sa(n, max_time):
        if n <= best_n:
            return None
            
        # Initialize random coloring
        # colors[i][j] for i < j
        colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]

        # Build masks
        # masks[v][c] is bitset of neighbors u such that color(v,u) == c
        masks = [[0] * 4 for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)

        # Calculate initial total triangles
        total_tri = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                total_tri += (masks[i][c] & masks[j][c]).bit_count()
        total_tri //= 3 # Each triangle counted 3 times

        if total_tri == 0:
            return n, colors

        T = 50.0
        cooling = 0.9999
        steps_since_improvement = 0
        best_total_tri = total_tri
        
        # Pre-compute list of edges for faster random access?
        # n=50, edges=1225. List is small.
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        num_edges = len(edges)

        while time.time() < max_time:
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            i, j = edges[idx]
            
            c_old = colors[i][j]
            # Pick a different color
            # random.randint(0, 3) might pick c_old
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue

            # Calculate delta triangles
            tri_old = (masks[i][c_old] & masks[j][c_old]).bit_count()
            tri_new = (masks[i][c_new] & masks[j][c_new]).bit_count()
            
            delta_tri = tri_new - tri_old
            
            if delta_tri <= 0 or random.random() < math.exp(-delta_tri / T):
                # Accept
                colors[i][j] = c_new
                colors[j][i] = c_new
                
                masks[i][c_old] &= ~(1 << j)
                masks[j][c_old] &= ~(1 << i)
                masks[i][c_new] |= (1 << j)
                masks[j][c_new] |= (1 << i)
                
                total_tri += delta_tri
                
                if total_tri < best_total_tri:
                    best_total_tri = total_tri
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if total_tri == 0:
                    return n, colors

                # Restart if stuck
                if steps_since_improvement > 100000:
                    # Re-initialize
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            colors[ii][jj] = random.randint(0, 3)
                            colors[jj][ii] = colors[ii][jj]
                    for ii in range(n):
                        for c in range(4):
                            masks[ii][c] = 0
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            c = colors[ii][jj]
                            masks[ii][c] |= (1 << jj)
                            masks[jj][c] |= (1 << ii)
                    total_tri = 0
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            c = colors[ii][jj]
                            total_tri += (masks[ii][c] & masks[jj][c]).bit_count()
                    total_tri //= 3
                    best_total_tri = total_tri
                    steps_since_improvement = 0
                    T = 50.0

            T *= cooling
            if T < 0.01:
                T = 0.01

        return None

    # Try to solve for descending n
    # Prioritize 50, then 49, etc.
    targets = [50, 49, 48, 45, 42]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
            
        # Allocate time: give 50s to 50, less to others
        if target_n == 50:
            time_limit = min(deadline - 5, time.time() + 50)
        elif target_n == 49:
            time_limit = min(deadline - 5, time.time() + 30)
        elif target_n == 48:
            time_limit = min(deadline - 5, time.time() + 15)
        else:
            time_limit = min(deadline - 5, time.time() + 5)

        result = solve_sa(target_n, time_limit)
        if result:
            n, colors = result
            out_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    out_colors.append(colors[i][j])
            return {'n': n, 'colors': out_colors}

    return best_result
