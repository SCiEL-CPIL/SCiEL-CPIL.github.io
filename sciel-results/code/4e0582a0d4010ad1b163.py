import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 50
    n_pairs = n // 2
    
    def get_color_diff(d, c):
        # d in 1..n-1
        if d == n // 2:
            return c[n // 2 - 1]
        else:
            return c[d - 1]

    def count_triangles(c):
        count = 0
        # Precompute color for each diff 1..n-1
        # diff_colors[0] is unused, diff_colors[1..n-1]
        # Actually, just use the list c directly with mapping
        # To speed up, create an array of length n
        dc = [0] * n
        for k in range(n // 2):
            # k is 0..24 for n=50
            # pair is k+1 and n-(k+1)
            val = c[k]
            d1 = k + 1
            d2 = n - d1
            dc[d1] = val
            dc[d2] = val
        # Note: for n=50, k=24 -> d1=25, d2=25. dc[25]=val. Correct.
        
        for a in range(1, n):
            ca = dc[a]
            for b in range(1, n):
                s = a + b
                if s == n:
                    continue
                c3 = n - s
                if c3 == 0:
                    continue
                cb = dc[b]
                cc3 = dc[c3]
                if ca == cb == cc3:
                    count += 1
        return count

    best_cost = float('inf')
    best_colors = None
    
    # Run SA
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    no_improve_counter = 0
    
    colors = [random.randint(0, 3) for _ in range(n_pairs)]
    current_cost = count_triangles(colors)
    best_cost = current_cost
    best_colors = colors[:]
    
    while time.time() < deadline:
        if current_cost == 0:
            break
            
        idx = random.randint(0, n_pairs - 1)
        old_color = colors[idx]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        colors[idx] = new_color
        new_cost = count_triangles(colors)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            colors[idx] = old_color
            no_improve_counter += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart if stuck
            if no_improve_counter > 1000:
                colors = [random.randint(0, 3) for _ in range(n_pairs)]
                current_cost = count_triangles(colors)
                T = 1.0
                no_improve_counter = 0

    if best_cost == 0 and best_colors is not None:
        colors_list = []
        # Precompute dc for best_colors
        dc = [0] * n
        for k in range(n // 2):
            val = best_colors[k]
            d1 = k + 1
            d2 = n - d1
            dc[d1] = val
            dc[d2] = val
            
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                colors_list.append(dc[d])
        return {'n': 50, 'colors': colors_list}
    else:
        # Fallback to n=32
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
