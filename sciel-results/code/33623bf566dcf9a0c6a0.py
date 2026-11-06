"""Program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 256
    import time
    import random

    def thue_morse(n):
        count = 0
        x = n
        while x:
            count += x & 1
            x >>= 1
        return count % 2

    # Initialize with Thue-Morse
    signs = np.array([(-1)**thue_morse(i) for i in range(1, n+1)], dtype=np.float64)

    def max_discrepancy_vec(signs):
        n = len(signs)
        max_val = 0.0
        for step in range(1, n + 1):
            max_len = (n + step - 1) // step
            rows = np.arange(step)
            cols = np.arange(max_len)
            indices = rows[:, None] + cols[None, :] * step
            valid_mask = indices < n
            
            mat = np.zeros((step, max_len), dtype=np.float64)
            mat[valid_mask] = signs[indices[valid_mask]]
            
            csums = np.cumsum(mat, axis=1)
            max_per_row = np.max(np.abs(csums), axis=1)
            current_step_max = np.max(max_per_row)
            
            if current_step_max > max_val:
                max_val = current_step_max
        return max_val

    current_max = max_discrepancy_vec(signs)
    best_max = current_max
    best_signs = signs.copy()
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 55.0
    
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, n-1)
        
        old_sign = signs[i]
        signs[i] = -old_sign
        
        new_max = max_discrepancy_vec(signs)
        
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            delta = new_max - current_max
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_max = new_max
                else:
                    signs[i] = old_sign
            else:
                # This case shouldn't happen if new_max >= current_max is checked above
                # but just in case of float issues
                signs[i] = old_sign
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    result = [int(s) for s in best_signs]
    assert len(result) == n
    assert all(s in (-1, 1) for s in result)
    return result
# EVOLVE-BLOCK-END
