import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Try to find a valid coloring for n=30, then 29, etc.
    # We know n=29 is possible. n=30 is possible (R=31).
    # We will attempt local search for n=30 and n=29.
    
    def count_mono_k222(colors, n):
        E = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                E[i, j] = E[j, i] = colors[idx]
                idx += 1
        
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        
        P = len(pairs)
        if P == 0:
            return 0
            
        M0 = np.zeros((P, P), dtype=np.uint8)
        M1 = np.zeros((P, P), dtype=np.uint8)
        
        # Optimize: Precompute pair indices for vertices?
        # Just double loop is fine for P~400.
        for p_idx in range(P):
            u, v = pairs[p_idx]
            for q_idx in range(P):
                if p_idx == q_idx:
                    continue
                x, y = pairs[q_idx]
                if u == x or u == y or v == x or v == y:
                    continue
                
                c1 = E[u, x]
                c2 = E[u, y]
                c3 = E[v, x]
                c4 = E[v, y]
                
                if c1 == c2 == c3 == c4:
                    if c1 == 0:
                        M0[p_idx, q_idx] = 1
                    else:
                        M1[p_idx, q_idx] = 1
        
        # Count triangles
        # Using matrix multiplication
        # t = trace(M^3) / 6
        # M @ M @ M
        t0 = np.trace(M0 @ M0 @ M0)
        t1 = np.trace(M1 @ M1 @ M1)
        return int((t0 + t1) / 6)

    def local_search(n, init_colors, max_steps=1000):
        colors = init_colors.copy()
        current_cost = count_mono_k222(colors, n)
        if current_cost == 0:
            return colors, 0
        
        best_colors = colors.copy()
        best_cost = current_cost
        T = 1.0
        cooling = 0.999
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random edge
            e_idx = random.randint(0, len(colors) - 1)
            old_color = colors[e_idx]
            new_color = 1 - old_color
            colors[e_idx] = new_color
            
            new_cost = count_mono_k222(colors, n)
            
            if new_cost == 0:
                return colors, 0
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
            else:
                colors[e_idx] = old_color
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_colors, best_cost

    # Try n=30 first
    for n in [30, 29, 28]:
        if time.time() > deadline - 10:
            break
            
        # Try Paley if n is prime and n % 4 == 1
        init = None
        if n in [29, 17, 13, 5]:
            squares = {x * x % n for x in range(1, n)}
            init = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
            
            # Check if Paley is valid
            cost = count_mono_k222(init, n)
            if cost == 0:
                return {'n': n, 'colors': init}
            
        # Try random init
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            if init is None:
                init = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            else:
                # Perturb init
                init = init.copy()
                for _ in range(n):
                    e_idx = random.randint(0, len(init) - 1)
                    init[e_idx] = 1 - init[e_idx]
            
            best_colors, best_cost = local_search(n, init, max_steps=500)
            if best_cost == 0:
                return {'n': n, 'colors': best_colors}
                
    # Fallback to n=17 Paley
    n = 17
    squares = {x * x % n for x in range(1, n)}
    return {'n': n, 'colors': [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]}
