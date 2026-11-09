import time
import random
import math

def construct():
    """
    Finds a 5-coloring of K_n with no monochromatic C4.
    Uses Simulated Annealing to minimize the number of monochromatic C4s.
    Returns the largest n for which a valid coloring is found and verified.
    """
    deadline = time.time() + 115  # Safety margin
    
    # We aim for n=26 first. If that fails, we can try smaller n, 
    # but SA usually finds 26 quickly if it's possible.
    # The Lazebnik-Woldar construction proves 26 is possible.
    
    target_n = 26
    best_n = 1
    best_colors = []
    
    # Try to find a valid coloring for target_n
    n = target_n
    colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
    
    # Helper to get/set color of edge (i, j)
    # Index: i * (n - 1) - i * (i + 1) // 2 + (j - i - 1) ? 
    # Simpler: precompute index map or use a 2D array for speed during SA, 
    # then flatten at the end.
    
    # Using a 2D list for colors is faster for SA updates.
    # colors_2d[i][j] for i < j.
    # We'll maintain a flat list for the final output, but use 2D for logic.
    
    def init_2d(n):
        c2d = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c2d[i][j] = random.randint(0, 4)
                c2d[j][i] = c2d[i][j]
        return c2d

    def count_c4s(c2d, n):
        # Count total monochromatic C4s
        # For each color c, count C4s.
        # A C4 is defined by 4 vertices. 
        # Efficient count: For each pair of vertices (u, v), count common neighbors in color c.
        # If count is k, it contributes to C4s?
        # No, a pair (u, v) with k common neighbors in color c forms k * (k-1) / 2 C4s?
        # Let's trace: u, v are opposite vertices in the C4.
        # If u and v share common neighbors x, y in color c, then u-x-v-y-u is a C4.
        # Yes. So for each color c, and each pair (u, v), let k = |N_c(u) \cap N_c(v)|.
        # The number of C4s where u and v are opposite is C(k, 2).
        # Summing this over all pairs (u, v) and all colors c gives the total number of monochromatic C4s.
        
        total = 0
        for c in range(5):
            # Build adjacency bitmasks for color c
            # adj_mask[i] is a bitmask of neighbors of i in color c
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if c2d[i][j] == c:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            # For each pair (u, v), count common neighbors
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[u] & adj_mask[v]
                    k = bin(common).count('1')
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def delta_c4(c2d, n, u, v, c_old, c_new):
        # Calculate change in C4 count when changing edge (u, v) from c_old to c_new
        # This is expensive if done naively.
        # However, we can approximate or use the full count if n is small enough.
        # For n=26, full count is O(5 * n^2 * n) = 5 * 26^3 ~ 88k ops per eval.
        # SA steps: 100k steps? 8.8e9 ops. Too slow in Python.
        
        # We need a faster delta calculation.
        # Change in C4s involving edge (u, v).
        # A C4 containing edge (u, v) in color c has the form u-a-v-b-u (edges (u,a),(a,v),(v,b),(b,u) all color c).
        # Wait, the edges are (u,a), (a,v), (v,b), (b,u).
        # So for a fixed (u, v) and color c, the number of such C4s is the number of pairs (a, b)
        # such that (u,a) is c, (a,b) is c, (b,v) is c.
        # Note: a and b must be distinct from u, v and each other.
        
        def count_c4s_with_edge(u, v, c):
            # Find all a such that (u, a) is c
            # Find all b such that (b, v) is c
            # Count pairs (a, b) such that (a, b) is c
            # a != v, b != u, a != b
            
            # Neighbors of u in color c: N_u
            # Neighbors of v in color c: N_v
            # We need a in N_u, b in N_v, and (a, b) is c.
            
            # Optimization: iterate over smaller set?
            # For n=26, sets are small.
            N_u = [x for x in range(n) if x != u and x != v and c2d[u][x] == c]
            N_v = [x for x in range(n) if x != u and x != v and c2d[v][x] == c]
            
            count = 0
            # For each a in N_u, for each b in N_v:
            # if a != b and c2d[a][b] == c: count++
            # To speed up, we can precompute for each a, the set of neighbors in color c.
            # But we are in a delta call, so c2d is changing.
            # Let's just do the double loop. Max size 24. 24*24 = 576 ops.
            for a in N_u:
                for b in N_v:
                    if a != b and c2d[a][b] == c:
                        count += 1
            return count

        # When we change (u, v) from c_old to c_new:
        # We lose C4s that used (u, v) in c_old.
        # We gain C4s that use (u, v) in c_new.
        
        # Note: The C4s counted by count_c4s_with_edge are those where (u, v) is an edge.
        # Does this cover all C4s affected?
        # Yes, a C4 is affected if and only if it contains the edge (u, v).
        
        c4_old = count_c4s_with_edge(u, v, c_old)
        c4_new = count_c4s_with_edge(u, v, c_new)
        
        # Wait, is this correct?
        # A C4 is a cycle of 4 edges. If (u, v) is an edge in the C4, then the other 3 edges are fixed.
        # The count_c4s_with_edge function counts the number of such C4s.
        # So Delta = c4_new - c4_old.
        
        return c4_new - c4_old

    # Simulated Annealing
    c2d = init_2d(n)
    current_energy = count_c4s(c2d, n)
    best_energy = current_energy
    best_c2d = [row[:] for row in c2d]
    
    T = 1.0
    cooling = 0.999
    
    # Number of steps
    # n=26, edges=325.
    # We want to run for ~100 seconds.
    # Each step involves:
    # 1. Pick random edge
    # 2. Pick random new color
    # 3. Calculate delta (fast)
    # 4. Update if accepted
    # 
    # Let's estimate speed.
    # count_c4s_with_edge is ~200-500 ops.
    # Python can do ~10M simple ops/sec.
    # So ~20k-50k steps/sec.
    # In 100s, we can do ~2-5 million steps.
    
    steps = 0
    max_steps = 5000000
    
    while time.time() < deadline and steps < max_steps:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        c_old = c2d[u][v]
        c_new = random.randint(0, 4)
        if c_new == c_old:
            continue
            
        dE = delta_c4(c2d, n, u, v, c_old, c_new)
        
        if dE < 0 or random.random() < math.exp(-dE / T):
            # Accept
            c2d[u][v] = c_new
            c2d[v][u] = c_new
            current_energy += dE
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_c2d = [row[:] for row in c2d]
                
            if best_energy == 0:
                # Found a valid coloring!
                break
                
        T *= cooling
        steps += 1
        
        # Occasionally re-evaluate full energy to correct drift?
        # Delta calculation should be exact, so drift shouldn't happen.
        # But just in case, every 10000 steps:
        if steps % 10000 == 0:
            current_energy = count_c4s(c2d, n)
            if current_energy < best_energy:
                best_energy = current_energy
                best_c2d = [row[:] for row in c2d]
            if best_energy == 0:
                break

    if best_energy == 0:
        # Flatten best_c2d to list
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(best_c2d[i][j])
        return {'n': n, 'colors': colors}
    else:
        # Failed to find valid coloring for n=26.
        # Try n=25? Or just return best found (which is invalid, so score 0).
        # Better to try smaller n if time permits.
        # But SA for 26 should work. If it doesn't, it might be due to bad random start.
        # Let's try a few restarts if we have time.
        
        if time.time() < deadline - 20:
            # Try n=25
            n = 25
            c2d = init_2d(n)
            current_energy = count_c4s(c2d, n)
            best_energy = current_energy
            best_c2d = [row[:] for row in c2d]
            T = 1.0
            cooling = 0.999
            steps = 0
            
            while time.time() < deadline - 5 and steps < max_steps:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                
                c_old = c2d[u][v]
                c_new = random.randint(0, 4)
                if c_new == c_old: continue
                
                dE = delta_c4(c2d, n, u, v, c_old, c_new)
                
                if dE < 0 or random.random() < math.exp(-dE / T):
                    c2d[u][v] = c_new
                    c2d[v][u] = c_new
                    current_energy += dE
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_c2d = [row[:] for row in c2d]
                    if best_energy == 0:
                        break
                T *= cooling
                steps += 1
                
                if steps % 10000 == 0:
                    current_energy = count_c4s(c2d, n)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_c2d = [row[:] for row in c2d]
                    if best_energy == 0:
                        break

            if best_energy == 0:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(best_c2d[i][j])
                return {'n': n, 'colors': colors}

        # Fallback to seed result if all fails (shouldn't happen)
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
