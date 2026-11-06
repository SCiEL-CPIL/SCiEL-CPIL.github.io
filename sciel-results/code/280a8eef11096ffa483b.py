import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute 4-APs
    # ap_list: list of (i1, i2, i3, i4)
    # neighbors[i]: list of (j1, j2, j3) where {i, j1, j2, j3} is a 4-AP
    
    ap_list = []
    for a in range(N):
        for d in range(1, N):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e < N:
                ap_list.append((a, b, c, e))
                
    neighbors = [[] for _ in range(N)]
    for ap in ap_list:
        a, b, c, e = ap
        neighbors[a].append((b, c, e))
        neighbors[b].append((a, c, e))
        neighbors[c].append((a, b, e))
        neighbors[e].append((a, b, c))
        
    # Convert to numpy arrays for faster access if needed, but lists of tuples might be faster in Python loops due to less overhead if N is small?
    # Let's stick to lists of lists of indices.
    
    def count_mono(x):
        cnt = 0
        for a, b, c, e in ap_list:
            if x[a] == x[b] == x[c] == x[e]:
                cnt += 1
        return cnt
        
    def delta_flip(i, x):
        # x is numpy array of 0s and 1s
        # i is index to flip
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for j1, j2, j3 in neighbors[i]:
            # Get values of other 3
            v1, v2, v3 = x[j1], x[j2], x[j3]
            
            # Check if mono before
            is_mono_before = (old_val == v1) and (old_val == v2) and (old_val == v3)
            
            # Check if mono after
            is_mono_after = (new_val == v1) and (new_val == v2) and (new_val == v3)
            
            if is_mono_before and not is_mono_after:
                delta -= 1
            elif not is_mono_before and is_mono_after:
                delta += 1
        return delta

    def greedy_search(x, time_limit):
        x = x.copy()
        current_cost = count_mono(x)
        best_cost = current_cost
        best_x = x.copy()
        
        while time.time() < time_limit:
            best_delta = 0
            best_i = -1
            
            # Find best flip
            # Optimization: Only check a subset? No, N=256 is small. Check all.
            for i in range(N):
                d = delta_flip(i, x)
                if d < best_delta:
                    best_delta = d
                    best_i = i
            
            if best_delta < 0:
                # Apply flip
                x[best_i] = 1 - x[best_i]
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            else:
                # Local minimum. Perturb.
                # Flip K random bits
                K = 5
                indices_to_flip = random.sample(range(N), K)
                for i in indices_to_flip:
                    x[i] = 1 - x[i]
                current_cost = count_mono(x) # Recalculate fully to be safe after multiple flips
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    
        return best_x, best_cost

    # Initializers
    initializers = []
    
    # 1. Random
    initializers.append(np.random.randint(0, 2, N))
    
    # 2. Thue-Morse
    tm = np.zeros(N, dtype=int)
    for i in range(N):
        # Count bits in binary representation of i
        tm[i] = bin(i).count('1') % 2
    initializers.append(tm)
    
    # 3. Periodic 0101...
    initializers.append(np.array([(i % 2) for i in range(N)]))
    
    # 4. Periodic 0011...
    initializers.append(np.array([(i // 2) % 2 for i in range(N)]))
    
    # 5. Random
    initializers.append(np.random.randint(0, 2, N))
    
    best_overall_cost = float('inf')
    best_overall_x = None
    
    # Allocate time per initializer
    time_per_init = (deadline - time.time()) / len(initializers)
    
    for i, init_x in enumerate(initializers):
        remaining_time = deadline - time.time()
        if remaining_time < 1.0: break
        
        # Give each initializer a fair share, but maybe weight them?
        # Just use equal shares for simplicity, or slightly more for Thue-Morse.
        t_limit = time.time() + remaining_time / (len(initializers) - i)
        
        res_x, res_cost = greedy_search(init_x, t_limit)
        
        if res_cost < best_overall_cost:
            best_overall_cost = res_cost
            best_overall_x = res_x
            
    # Final check
    if best_overall_x is None:
        best_overall_x = initializers[0]
        
    return [int(v) for v in best_overall_x]
