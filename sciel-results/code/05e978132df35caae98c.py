import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, 20s buffer
    
    # We want to find M with many k such that k*sigma(k)=M.
    # We iterate k up to K_max.
    # K_max should be large enough to find collisions.
    # M = k*sigma(k) ~ k^2 * ln(ln(k)).
    # If M <= 10^18, k can be up to ~ 10^9.
    # But we are limited by time and memory.
    
    # Let's try K_max = 2 * 10^7 first. It's fast.
    # If time remains, we can try larger.
    # But a single large run is better than many small ones if we can find the global max.
    # Collisions are rare for large k. Most collisions happen for small k.
    # For example, 336, 372, 434 are small.
    # So focusing on small k is good.
    
    K_max = 20_000_000 # 20 million
    
    # Compute sigma(k) for k = 1..K_max
    # sigma = np.zeros(K_max + 1, dtype=np.int64)
    # for d in range(1, K_max + 1):
    #     sigma[d::d] += d
    
    # This loop is O(N log N).
    # N=20M, log N ~ 17. Ops ~ 340M. NumPy should handle this in < 5s.
    
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    
    # Optimization: We can break early if time is up
    for d in range(1, K_max + 1):
        if time.time() > deadline - 5.0:
            break
        sigma[d::d] += d
        
    # Calculate M = k * sigma(k)
    # k is 1..K_max
    k_arr = np.arange(1, K_max + 1, dtype=np.int64)
    M_arr = k_arr * sigma[1:]
    
    # Filter M <= 10^18
    # Note: sigma(k) can be large, but for k=20M, sigma(k) < 100M, M < 2*10^12.
    # So all M are < 10^18.
    valid_mask = M_arr <= 10**18
    
    # Group k by M
    # We need to find the M with the most k's.
    # Using a dictionary.
    # To avoid overhead of converting numpy arrays to lists for each M,
    # we can sort M_arr and k_arr together?
    # Or just iterate.
    
    # Iterating 20M items in Python is slow (~1s).
    # Let's use a more efficient grouping.
    # Since M values are not sparse, we can use a dict.
    
    solutions = {}
    
    # To speed up, we can check time periodically
    # But 20M iterations in Python is ~1-2s.
    # Let's just do it.
    
    # Actually, we can use pandas or just numpy unique?
    # np.unique returns unique values and indices.
    # unique_M, indices = np.unique(M_arr[valid_mask], return_inverse=True)
    # Then we can count occurrences of each unique_M.
    # And then find the max count.
    # Then retrieve the k's for that M.
    
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    # Sort k_valid by M_valid?
    # It's easier to use a dictionary.
    # But let's try the unique approach.
    
    # np.unique is O(N log N). 20M log 20M is fast in C.
    unique_M, inverse = np.unique(M_valid, return_inverse=True)
    
    # Count occurrences
    counts = np.bincount(inverse)
    
    # Find index of max count
    max_idx = np.argmax(counts)
    max_M = unique_M[max_idx]
    max_count = counts[max_idx]
    
    # Retrieve k's for max_M
    # We need k_valid where M_valid == max_M
    # We can use the inverse array? No, inverse maps to unique_M index.
    # So we need k_valid[inverse == max_idx]
    
    k_solutions = k_valid[inverse == max_idx]
    
    # Convert to list of Python ints
    k_list = k_solutions.tolist()
    
    # Verify
    if not k_list:
        return {'M': 1, 'k': [1]}
        
    return {'M': int(max_M), 'k': k_list}
