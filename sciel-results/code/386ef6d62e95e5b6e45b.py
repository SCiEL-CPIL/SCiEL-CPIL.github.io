import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 20s safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # N_limits to try
    # We increase N to search for more collisions
    N_limits = [10**6, 10**7, 2 * 10**7, 5 * 10**7]
    
    for N in N_limits:
        if time.time() > deadline - 30:  # Reserve 30s for processing and safety
            break
        
        # Compute sigma up to N
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Optimization: Split the loop
        # For i <= N/2, sigma[i::i] has length > 1
        # For i > N/2, sigma[i::i] has length 1 (just sigma[i])
        half = N // 2
        
        # Loop for i <= N/2
        for i in range(1, half + 1):
            sigma[i::i] += i
            
        # Vectorized update for i > N/2
        if half < N:
            # Indices from half+1 to N
            # sigma[half+1:] += np.arange(half+1, N+1)
            sigma[half+1:] += np.arange(half + 1, N + 1, dtype=np.int64)
        
        # Compute M = k * sigma(k) for k=1..N
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:]
        
        # Filter M <= 10**18
        mask = M_vals <= 10**18
        M_vals = M_vals[mask]
        k_vals = k_vals[mask]
        
        if len(M_vals) == 0:
            continue
        
        # Find duplicates
        # np.unique is efficient for large arrays
        unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
        
        # Find max count
        max_count = np.max(counts)
        
        if max_count > best_count:
            # Find the M with max count
            # If multiple, pick the first one
            max_indices = np.where(counts == max_count)[0]
            idx = max_indices[0]
            
            best_M = int(unique_M[idx])
            best_count = int(max_count)
            
            # Retrieve k's
            mask_k = inverse == idx
            best_ks = k_vals[mask_k].tolist()
            
            # Check time
            if time.time() > deadline - 10:
                break
        else:
            # If no improvement, we can potentially skip further N if time is tight
            # But larger N might find new M's with higher counts
            # So we continue if time permits
            pass

    return {'M': best_M, 'k': best_ks}
