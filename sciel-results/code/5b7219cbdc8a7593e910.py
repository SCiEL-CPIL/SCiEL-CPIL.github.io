import time
import random
import math

def construct():
    deadline = time.time() + 140.0
    
    def count_mono_triangles(colors, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[i][j]
                for k in range(j + 1, n):
                    if c_ij == colors[i][k] == colors[j][k]:
                        count += 1
        return count

    def find_valid_coloring(n, time_budget):
        """
        Find a 4-coloring of K_n with no mono triangles.
        Returns colors matrix or None.
        """
        start_time = time.time()
        
        # Initialize random
        colors = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            colors[i][i] = -1 
            
        # Initial cost
        current_cost = count_mono_triangles(colors, n)
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = [row[:] for row in colors]
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        
        while time.time() - start_time < time_budget:
            if T < 1e-4:
                T = 1.0
                # Restart if stuck?
                if random.random() < 0.1:
                    colors = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
                    for i in range(n):
                        colors[i][i] = -1
                    current_cost = count_mono_triangles(colors, n)
                    if current_cost == 0:
                        return colors
                    best_cost = current_cost
                    best_colors = [row[:] for row in colors]
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            old_c = colors[u][v]
            
            # Try all other colors
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                delta = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    a = colors[v][w]
                    b = colors[w][u]
                    
                    # Check if triangle (u,v,w) becomes monochromatic
                    # Old: color(u,v)=old_c, color(v,w)=a, color(w,u)=b
                    # Mono if old_c == a == b
                    old_mono = 1 if (old_c == a == b) else 0
                    # New: color(u,v)=new_c
                    new_mono = 1 if (new_c == a == b) else 0
                    delta += new_mono - old_mono
                    
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            # Acceptance
            if best_new_c == -1:
                continue
                
            if best_delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-best_delta / T)
                
            if accept:
                colors[u][v] = best_new_c
                colors[v][u] = best_new_c
                current_cost += best_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = [row[:] for row in colors]
                    if best_cost == 0:
                        return best_colors
                        
            T *= cooling
            
        return None if best_cost > 0 else best_colors

    # Try n from 50 down to 33
    target_ns = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    times = {
        50: 30.0,
        49: 20.0,
        48: 15.0,
        47: 10.0,
        46: 8.0,
        45: 5.0,
        44: 4.0,
        43: 3.0,
        42: 2.0,
        41: 1.0,
        40: 1.0,
        39: 1.0,
        38: 1.0,
        37: 1.0,
        36: 1.0,
        35: 1.0,
        34: 1.0,
        33: 1.0
    }
    
    def build_seed_32():
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
        k16_colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                k16_colors.append(log[i ^ j] % 3)
        
        def base_to_2d(base_colors, m):
            mat = [[-1] * m for _ in range(m)]
            idx = 0
            for i in range(m):
                for j in range(i + 1, m):
                    mat[i][j] = mat[j][i] = base_colors[idx]
                    idx += 1
            return mat
        
        def double_coloring_from_mat(mat, m):
            n = 2 * m
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    if i // m == j // m:
                        colors.append(mat[i % m][j % m])
                    else:
                        colors.append(3)
            return colors
        
        k16_mat = base_to_2d(k16_colors, 16)
        return double_coloring_from_mat(k16_mat, 16)

    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
        t_budget = times.get(n, 1.0)
        if time.time() + t_budget > deadline - 2.0:
            t_budget = max(0.1, deadline - 2.0 - time.time())
        
        res = find_valid_coloring(n, t_budget)
        if res is not None:
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors_list.append(res[i][j])
            return {'n': n, 'colors': colors_list}
            
    seed_colors = build_seed_32()
    return {'n': 32, 'colors': seed_colors}
