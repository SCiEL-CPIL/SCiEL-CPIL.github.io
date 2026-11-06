import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Use a list for speed in Python for scalar updates
    lpf = [0] * (MAX_VAL + 1)
    
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            for multiple in range(p, MAX_VAL + 1, p):
                lpf[multiple] = p
                
    if time.time() > deadline:
        return LO

    # Convert to numpy for fast vectorized max and argmin
    lpf_arr = np.array(lpf, dtype=np.int32)
    
    lpf_n = lpf_arr[LO:HI+1]
    lpf_n1 = lpf_arr[LO+1:HI+2]
    
    costs = np.maximum(lpf_n, lpf_n1)
    min_idx = np.argmin(costs)
    best_n = LO + min_idx
    
    return int(best_n)
