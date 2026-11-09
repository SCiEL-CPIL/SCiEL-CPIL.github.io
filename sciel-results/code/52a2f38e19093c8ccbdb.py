import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Choose N. 4*10^7 should be safe.
    N = 40_000_000
    
    sigma = np.zeros(N+1, dtype=np.int64)
    # Sieve for sigma
    for i in range(1, N+1):
        sigma[i::i] += i
        
    ks = np.arange(1, N+1, dtype=np.int64)
    Ms = ks * sigma
    
    # Filter M <= 10^18 (should be all for this N)
    limit = 10**18
    mask = Ms <= limit
    valid_Ms = Ms[mask]
    valid_ks = ks[mask]
    
    # Find most frequent M
    unique_Ms, counts, inverse = np.unique(valid_Ms, return_counts=True, return_inverse=True)
    max_count = np.max(counts)
    max_idx = np.argmax(counts) # Gets first max
    
    best_M = unique_Ms[max_idx]
    k_vals = valid_ks[inverse == max_idx]
    
    return {'M': int(best_M), 'k': [int(k) for k in k_vals]}
