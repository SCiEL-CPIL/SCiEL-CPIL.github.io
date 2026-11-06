"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math

def construct():
    n = 256
    
    # Fast discrepancy calculation using NumPy
    def calc_disc(a):
        a = np.asarray(a)
        max_abs = 0
        for d in range(1, n + 1):
            for r in range(d):
                # Extract the arithmetic progression
                # a[r::d]
                seq = a[r::d]
                if len(seq) == 0:
                    continue
                # Prefix sums
                # We need P[0]=0, P[1]=a[0], P[2]=a[0]+a[1]...
                # np.cumsum gives P[1:]
                cs = np.concatenate(([0], np.cumsum(seq)))
                # The max absolute sum of any subarray is max(cs) - min(cs)
                val = np.max(cs) - np.min(cs)
                if val > max_abs:
                    max_abs = val
        return int(max_abs)

    # Initialize with Thue-Morse
    signs = []
    for i in range(n):
        c = bin(i).count('1')
        signs.append(1 if c % 2 == 0 else -1)
    
    current_signs = np.array(signs, dtype=np.int8)
    current_disc = calc_disc(current_signs)
    best_signs = current_signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing
    T = 5.0
    T_end = 0.01
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 55.0
    
    # Precompute indices for faster access? Not needed for numpy slicing.
    
    while time.time() - start_time < time_limit:
        # Flip a random bit
        idx = rng.randint(0, n - 1)
        old_val = current_signs[idx]
        current_signs[idx] = -old_val
        
        new_disc = calc_disc(current_signs)
        
        delta = new_disc - current_disc
        
        if delta < 0:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current_signs.copy()
        else:
            # Metropolis criterion
            if T > 0:
                # Avoid overflow in exp
                if delta / T < 50: 
                    prob = math.exp(-delta / T)
                else:
                    prob = 0
                if rng.random() < prob:
                    current_disc = new_disc
                else:
                    # Revert
                    current_signs[idx] = old_val
        
        # Cool down
        T *= 0.9998 
        if T < T_end:
            T = T_end
            
    return list(best_signs)

# EVOLVE-BLOCK-END
