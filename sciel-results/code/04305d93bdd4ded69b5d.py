import time
import math

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Primary approach: SPF sieve guarantees finding the global minimum
    max_val = N_MAX + 1
    lpf_arr = [1] * (max_val + 1)
    
    for i in range(2, max_val + 1):
        if lpf_arr[i] == 1:
            for j in range(i, max_val + 1, i):
                lpf_arr[j] = i
    
    best_n = N_MIN
    best_lpf = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        lpf_n = lpf_arr[n]
        lpf_np1 = lpf_arr[n + 1]
        combined_lpf = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if combined_lpf < best_lpf:
            best_lpf = combined_lpf
            best_n = n
        
        if time.time() - start_time > deadline:
            break
    
    return best_n
