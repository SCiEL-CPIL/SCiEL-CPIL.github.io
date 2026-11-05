import time
import random
import math

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    best_n = 32
    best_colors = None
    
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
    best_colors = seed_colors

    def solve_for_n(n, time_limit):
        m = n * (n - 1) // 2
        
        # Precompute edge index mapping
        idx = [[0]*n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1

        def local_search():
            colors = [random.randint(0, 3) for _ in range(m)]
            
            masks = [[0] * 4 for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)

            tri_count = 0
            for u in range(n):
                for v in range(u+1, n):
                    c = colors[idx[u][v]]
                    tri_count += (masks[u][c] & masks[v][c]).bit_count()
            
            tri_count //= 3

            if tri_count == 0:
                return colors

            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            # Use a faster random number generator if possible? random.randint is okay.
            # Optimize the inner loop
            
            while T > min_T and time.time() < time_limit:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                i_idx = idx[u][v]
                old_c = colors[i_idx]
                new_c = random.randint(0, 3)
                if old_c == new_c:
                    continue
                
                lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
                gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
                
                delta = gained - lost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    colors[i_idx] = new_c
                    tri_count += delta
                    
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    
                    if tri_count == 0:
                        return colors
                    
                T *= cooling

            return None

        # Multi-restart
        while time.time() < time_limit:
            res = local_search()
            if res is not None:
                return res
        
        return None

    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 10:
            break
            
        # Allocate 95% of remaining time to this n
        search_time = time_left * 0.95
        res = solve_for_n(n, time.time() + search_time)
        
        if res is not None:
            # Verify
            valid = True
            idx = [[0]*n for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    idx[i][j] = k
                    idx[j][i] = k
                    k += 1
            for u in range(n):
                for v in range(u+1, n):
                    for w in range(v+1, n):
                        if res[idx[u][v]] == res[idx[v][w]] == res[idx[w][u]]:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return {'n': n, 'colors': res}
            
    return {'n': best_n, 'colors': best_colors}
