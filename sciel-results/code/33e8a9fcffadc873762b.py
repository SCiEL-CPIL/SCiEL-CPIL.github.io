import math
import numpy as np

def construct():
    # Use NumPy for efficient sigma calculation and collision detection
    # Search space N=5,000,000 allows finding M with more solutions
    N = 5000000
    
    # Sieve for sigma function
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Calculate M = k * sigma(k) for k in 1..N
    k_arr = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_arr * sigma[1:]
    
    # Filter M <= 10^18
    max_M = 10**18
    mask = M_vals <= max_M
    valid_M = M_vals[mask]
    valid_k = k_arr[mask]
    
    if len(valid_M) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find M with the most solutions (highest frequency)
    unique_M, counts = np.unique(valid_M, return_counts=True)
    
    max_count = np.max(counts)
    
    # If max count is not better than seed (3), return seed
    if max_count <= 3:
         return {'M': 333312, 'k': [336, 372, 434]}
         
    # Find the first M with this max count
    idx = np.argmax(counts)
    best_M = int(unique_M[idx])
    
    # Retrieve all k corresponding to best_M
    # This creates a boolean mask of size ~5M, which is fine
    best_k = valid_k[valid_M == best_M]
    
    # Convert to Python int list
    best_k_list = [int(x) for x in best_k]
    
    return {'M': best_M, 'k': best_k_list}
