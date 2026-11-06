import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Seed solution n=9
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = seed_colors
    
    for n_target in [17, 18, 19]:
        if time.time() > deadline - 10.0:
            break
        
        result = run_sa(n_target, deadline - 5.0)
        
        if result is not None:
            best_n = n_target
            best_colors = result
            
    return {'n': best_n, 'colors': best_colors}

def run_sa(n, deadline):
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    masks = [[0] * n for _ in range(4)]
    
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            idx_map[(i,j)] = idx
            idx += 1
            
    def get_idx(u, v):
        if u > v: u, v = v, u
        return idx_map[(u,v)]
        
    for i in range(n):
        for j in range(i+1, n):
            c = colors[get_idx(i,j)]
            masks[c][i] |= (1 << j)
            masks[c][j] |= (1 << i)
            
    def count_paths_len4(u, v, c):
        N_u = masks[c][u]
        N_v = masks[c][v]
        
        sum_prod = 0
        for y in range(n):
            if y == u or y == v:
                continue
            # Count neighbors of y in N_u and N_v
            # This is equivalent to popcount(mask[y] & N_u)
            a_y = bin(masks[c][y] & N_u).count('1')
            b_y = bin(masks[c][y] & N_v).count('1')
            sum_prod += a_y * b_y
            
        return sum_prod

    def get_energy():
        e = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[get_idx(i,j)]
                e += count_paths_len4(i, j, c)
        return e // 6

    current_energy = get_energy()
    best_energy = current_energy
    best_state = colors[:]
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n-1)
        j = random.randint(i+1, n-1)
        idx = get_idx(i, j)
        c_old = colors[idx]
        c_new = random.choice([c for c in range(4) if c != c_old])
        
        delta = 0
        delta -= count_paths_len4(i, j, c_old)
        delta += count_paths_len4(i, j, c_new)
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colors[idx] = c_new
            
            masks[c_old][i] &= ~(1 << j)
            masks[c_old][j] &= ~(1 << i)
            masks[c_new][i] |= (1 << j)
            masks[c_new][j] |= (1 << i)
            
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = colors[:]
                if best_energy == 0:
                    return best_state
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
    return best_state if best_energy == 0 else None
