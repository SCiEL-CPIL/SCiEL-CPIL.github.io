import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    # Initial random sequence
    s = [random.choice([-1, 1]) for _ in range(N)]
    
    def evaluate(s):
        max_abs = 0
        worst_a, worst_d, worst_len = 0, 1, 1 # dummy
        # We store the worst AP found to guide search
        # Actually, returning the worst AP details is expensive if we just return the value.
        # Let's return value and the indices of the worst AP.
        
        for d in range(1, N + 1):
            for r in range(d):
                # Compute max subarray sum and min subarray sum for s[r], s[r+d], ...
                c_max = 0
                c_min = 0
                max_sub = 0
                min_sub = 0
                start_max = r
                end_max = r
                cur_start_max = r
                
                start_min = r
                end_min = r
                cur_start_min = r
                
                # We need to track the actual indices for the worst AP
                # This adds overhead. For now, just get the value.
                # If we want to guide, we can re-scan for the worst AP only when we improve,
                # or just pick random indices.
                # Given N=256, re-scanning is cheap.
                
                for j in range(r, N, d):
                    val = s[j]
                    
                    # Max Kadane
                    if c_max + val < val:
                        c_max = val
                        cur_start_max = j
                    else:
                        c_max += val
                    if c_max > max_sub:
                        max_sub = c_max
                        start_max = cur_start_max
                        end_max = j
                        
                    # Min Kadane
                    if c_min + val > val:
                        c_min = val
                        cur_start_min = j
                    else:
                        c_min += val
                    if c_min < min_sub:
                        min_sub = c_min
                        start_min = cur_start_min
                        end_min = j

                # Check if this AP is worse than current global max
                if max_sub > max_abs:
                    max_abs = max_sub
                    worst_a = start_max
                    worst_d = d
                    worst_len = (end_max - start_max) // d + 1
                    worst_sign = 1 # It's a positive sum
                if -min_sub > max_abs:
                    max_abs = -min_sub
                    worst_a = start_min
                    worst_d = d
                    worst_len = (end_min - start_min) // d + 1
                    worst_sign = -1 # It's a negative sum (so flipping signs to positive helps)
                    
        return max_abs, worst_a, worst_d, worst_len, worst_sign

    current_val, worst_a, worst_d, worst_len, worst_sign = evaluate(s)
    
    best_val = current_val
    best_s = s[:]
    
    while time.time() - start_time < timeout:
        # Determine candidate indices to flip
        # With high probability, flip an index in the worst AP
        if random.random() < 0.7:
            # Pick random index in the worst AP
            # Indices: worst_a, worst_a + worst_d, ..., worst_a + (worst_len-1)*worst_d
            k = random.randint(0, worst_len - 1)
            idx = worst_a + k * worst_d
        else:
            idx = random.randint(0, N - 1)
            
        # Flip
        s[idx] *= -1
        
        new_val, new_wa, new_wd, new_wl, new_ws = evaluate(s)
        
        if new_val < best_val:
            best_val = new_val
            best_s = s[:]
            current_val = new_val
            worst_a, worst_d, worst_len, worst_sign = new_wa, new_wd, new_wl, new_ws
        elif new_val == current_val:
            # Accept to explore
            current_val = new_val
            worst_a, worst_d, worst_len, worst_sign = new_wa, new_wd, new_wl, new_ws
        else:
            # Revert
            s[idx] *= -1
            
    return best_s
