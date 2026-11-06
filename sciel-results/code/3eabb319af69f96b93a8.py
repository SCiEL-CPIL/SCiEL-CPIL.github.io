"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    N = 1000000 + 24  # max index needed: n+24 with n up to 1e6
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute 23-smooth part for 0..N using NumPy
    sf = np.arange(N + 1, dtype=np.int64)
    for p in primes:
        pk = p
        while pk <= N:
            sf[pk::pk] //= p
            pk *= p
    
    # Check deadline
    if time.time() - start > DEADLINE:
        return 1000
    
    # Convert to list for faster indexing in the sliding window
    sf_list = sf.tolist()
    
    # Sliding window to find n in [1000,1000000] minimizing distinct count
    best_n = 1000
    best_count = float('inf')
    
    # Initialize window for n=1000: indices 1001..10024
    win = Counter(sf_list[1001:10025])
    distinct = len(win)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    for n in range(1001, 1000001):
        old_idx = n
        new_idx = n + 24
        c_old = win[sf_list[old_idx]]
        c_old -= 1
        if c_old == 0:
            del win[sf_list[old_idx]]
        else:
            win[sf_list[old_idx]] = c_old
        
        c_new = win.get(sf_list[new_idx], 0)
        win[sf_list[new_idx]] = c_new + 1
        
        distinct = len(win)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break
        
        if time.time() - start > DEADLINE:
            break
    
    return best_n
# EVOLVE-BLOCK-END
