import numpy as np
import time
import random

N = 256

def _count_mono4(c):
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0: break
        # Vectorized check for all a
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _delta_flip(c, i):
    # c is numpy array of int8
    v = c[i]
    new_v = 1 - v
    delta = 0
    # i is 0-indexed.
    # AP: a, a+d, a+2d, a+3d
    # i is one of these.
    # k=0: i=a -> a=i
    # k=1: i=a+d -> a=i-d
    # k=2: i=a+2d -> a=i-2d
    # k=3: i=a+3d -> a=i-3d
    
    max_d = N // 3
    for d in range(1, max_d + 1):
        for k in range(4):
            a = i - k * d
            if a < 0: continue
            if a + 3 * d >= N: continue
            
            # The other 3 indices
            idxs = [a, a+d, a+2*d, a+3*d]
            # Remove i
            others = [idxs[j] for j in range(4) if idxs[j] != i]
            
            # Check if others are all v (was mono)
            was_mono = True
            is_mono = True
            for j in others:
                if c[j] != v: was_mono = False
                if c[j] != new_v: is_mono = False
            
            if was_mono: delta -= 1
            if is_mono: delta += 1
    return delta

def _local_search(c, time_limit):
    start_time = time.time()
    C = _count_mono4(c)
    best_c = c.copy()
    best_C = C
    
    while time.time() - start_time < time_limit:
        improved = False
        # Random order or specific order? Random helps escape symmetry.
        indices = list(range(N))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit - 5: break # safety margin
            
            dC = _delta_flip(c, i)
            if dC < 0:
                c[i] = 1 - c[i]
                C += dC
                improved = True
                if C < best_C:
                    best_C = C
                    best_c = c.copy()
        
        if not improved:
            # Stuck in local minimum. 
            # Try a random kick? Or just break if we want pure hill climbing.
            # Let's try a few random flips to escape.
            for _ in range(10):
                i = random.randint(0, N-1)
                c[i] = 1 - c[i]
            # Continue search from here? Or restart?
            # Restarting from best_c is safer.
            c = best_c.copy()
            C = best_C
            # Maybe add noise to best_c?
            # Let's just break and rely on multiple starts in construct()
            break
            
    return best_c, best_C

def construct():
    start_time = time.time()
    deadline = 110.0 # Keep 10s buffer
    
    # Initial guess: Thue-Morse
    c = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
    
    best_c = c.copy()
    best_C = _count_mono4(best_c)
    
    # Run local search
    time_per_run = 20.0
    runs = 0
    while time.time() - start_time < deadline - time_per_run:
        # Perturb the current best? Or start fresh?
        # Starting fresh from Thue-Morse is good.
        # Maybe try a random start occasionally?
        if runs % 5 == 0:
            c = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
        else:
            c = best_c.copy()
            # Perturb
            for _ in range(5):
                idx = random.randint(0, N-1)
                c[idx] = 1 - c[idx]
                
        c, C = _local_search(c, time_per_run)
        
        if C < best_C:
            best_C = C
            best_c = c.copy()
            
        runs += 1
        if time.time() - start_time > deadline:
            break
            
    # Final polish with remaining time
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        c, C = _local_search(best_c, remaining)
        if C < best_C:
            best_C = C
            best_c = c.copy()

    # Convert to list of ints 0/1
    return [int(x) for x in best_c]
