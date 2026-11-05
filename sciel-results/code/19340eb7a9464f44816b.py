import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit
    
    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
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
    
    def count_mono_triangles(adj, n):
        """Count total monochromatic triangles using bitmasks."""
        total = 0
        for k in range(4):
            for i in range(n):
                m = adj[k][i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    if j > i:
                        common = adj[k][i] & adj[k][j]
                        total += common.bit_count()
                    m ^= lb
        return total // 3
    
    def greedy_init(n, rng):
        """Greedy initialization that tries to avoid triangles."""
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    common = adj[k][i] & adj[k][j]
                    delta = common.bit_count()
                    if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                        best_delta = delta
                        best_k = k
                
                color[i][j] = best_k
                color[j][i] = best_k
                adj[best_k][i] |= (1 << j)
                adj[best_k][j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)
    
    def random_init(n, rng):
        """Random initialization."""
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                k = rng.randint(0, 3)
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)
    
    def run_local_search(adj, color, n, time_limit, rng, T_init=2.0, T_min=0.1, cooling_rate=0.9999):
        """Local search with simulated annealing to minimize monochromatic triangles."""
        current_score = count_mono_triangles(adj, n)
        if current_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return adj, color, 0
        
        best_score = current_score
        best_adj = [row[:] for row in adj]
        best_color = [row[:] for row in color]
        
        T = T_init
        
        while time.time() < time_limit:
            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_k = color[i][j]
            new_k = rng.randint(0, 3)
            if new_k == old_k:
                continue
            
            # Calculate delta in mono triangles
            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()
            
            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()
            
            delta = delta_add - delta_remove
            
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    best_color = [row[:] for row in color]
                    if best_score == 0:
                        colors = []
                        for i2 in range(n):
                            for j2 in range(i2 + 1, n):
                                colors.append(best_color[i2][j2])
                        return best_adj, best_color, 0
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_adj, best_color, best_score
    
    def try_solve_n(n, time_limit, rng):
        """Try to find a valid coloring for K_n using multiple strategies."""
        # Try greedy init first
        adj, color, score = greedy_init(n, rng)
        if score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return {'n': n, 'colors': colors}
        
        # Run local search from greedy init
        best_adj, best_color, best_score = run_local_search(adj, color, n, time_limit / 2, rng)
        
        if best_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(best_color[i][j])
            return {'n': n, 'colors': colors}
        
        # Try random init if time remains
        remaining = time_limit - (time.time() - (time_limit - time_limit))
        if time.time() < time_limit - 1:
            adj2, color2, score2 = random_init(n, rng)
            if score2 < best_score:
                best_adj, best_color, best_score = run_local_search(adj2, color2, n, time.time() < time_limit - 1 and time_limit - time.time() or 0.1, rng)
        
        if best_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(best_color[i][j])
            return {'n': n, 'colors': colors}
        
        return None
    
    # Try to find solutions for n=50 down to 33
    rng = random.Random(42)
    
    for n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]:
        if time.time() > deadline - 3:
            break
        
        remaining = deadline - time.time() - 1
        if remaining < 2:
            break
        
        # Allocate more time for larger n
        time_per_n = min(remaining * 0.9, 15.0) if n >= 45 else min(remaining * 0.9, 8.0)
        
        res = try_solve_n(n, time_per_n, rng)
        if res:
            return res
    
    return get_fallback()
