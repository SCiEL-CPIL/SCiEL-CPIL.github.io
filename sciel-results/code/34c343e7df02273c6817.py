import math
import numpy as np

def construct():
    N = 10**7
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    k_arr = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_arr * sigma[1:]
    
    mask = M_vals <= 10**18
    M_valid = M_vals[mask]
    k_valid = k_arr[mask]
    
    if len(M_valid) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
    
    max_count = np.max(counts)
    
    if max_count <= 5:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    best_idx = np.argmax(counts)
    best_M = unique_M[best_idx]
    
    k_list = k_valid[inverse == best_idx]
    
    k_py = [int(x) for x in k_list]
    
    if len(k_py) > 1000:
        k_py = k_py[:1000]
        
    return {'M': int(best_M), 'k': k_py}
