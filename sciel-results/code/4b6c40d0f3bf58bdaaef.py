import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # We can try to increase N if we have time, but let's start with 10^7
    N = 10**7
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # Check time
    if time.time() > deadline:
        # This shouldn't happen if N is small enough
        # But if it does, we return what we have? 
        # No, the loop is done.
        pass
        
    ks_arr = np.arange(1, N + 1, dtype=np.int64)
    vals = ks_arr * sigma
    
    max_M_limit = 10**18
    valid_mask = vals <= max_M_limit
    if not np.all(valid_mask):
        vals = vals[valid_mask]
        ks_arr = ks_arr[valid_mask]
        
    if len(vals) == 0:
        return {'M': 1, 'k': [1]}
        
    unique_vals, counts = np.unique(vals, return_counts=True)
    
    max_count = np.max(counts)
    best_M_idx = np.argmax(counts)
    best_M = unique_vals[best_M_idx]
    
    mask = vals == best_M
    ks = ks_arr[mask]
    
    k_list = [int(x) for x in ks]
    
    return {'M': int(best_M), 'k': k_list}
