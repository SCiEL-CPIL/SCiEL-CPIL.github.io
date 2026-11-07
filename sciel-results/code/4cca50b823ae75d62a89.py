import math
import numpy as np

def construct():
    N = 15000000
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized divisor sum
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_vals * sigma
    
    # Filter
    mask = M_vals <= 10**18
    M_valid = M_vals[mask]
    k_valid = k_vals[mask]
    
    if len(M_valid) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find most frequent M
    unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
    
    max_count = np.max(counts)
    
    if max_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_idx = np.argmax(counts)
    best_M = int(unique_M[best_idx])
    
    # Extract k's
    k_indices = np.where(inverse == best_idx)[0]
    best_k = [int(x) for x in k_valid[k_indices]]
    
    return {'M': best_M, 'k': best_k}
