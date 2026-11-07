import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Choose K_max
    # 20 million is a good balance of memory and search depth
    K_max = 20_000_000
    
    # Initialize sigma array
    # Using int64 to avoid overflow (though M < 10^18, sigma(k) is small)
    sig = np.zeros(K_max + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This loop runs K_max times
    # Each iteration performs a slice addition
    for i in range(1, K_max + 1):
        sig[i::i] += i
        
        # Check time periodically (every 1 million iterations)
        if i % 1_000_000 == 0:
            if time.time() > deadline - 20:
                # If we are running out of time, we can break
                # But we need to adjust K_max to i to ensure sig is valid up to i
                # Actually, if we break, sig[j] for j > i might be incomplete.
                # So we should only use k <= i.
                K_max = i
                break
    
    # Generate k values
    ks = np.arange(1, K_max + 1, dtype=np.int64)
    
    # Calculate M = k * sigma(k)
    Ms = ks * sig
    
    # Filter M <= 10^18
    # For K=20M, max M is ~10^15, so this filter is likely a no-op
    # but good for safety if we increase K significantly
    valid_mask = Ms <= 10**18
    Ms = Ms[valid_mask]
    ks = ks[valid_mask]
    
    # Find the M with the most frequent occurrences
    # np.unique sorts the array, which is O(N log N)
    unique_Ms, counts, inverse = np.unique(Ms, return_counts=True, return_inverse=True)
    
    # Handle case where no valid M found (should not happen)
    if len(unique_Ms) == 0:
        return {'M': 1, 'k': [1]}
        
    # Find the index of the M with the maximum count
    max_count = np.max(counts)
    best_M_idx = np.argmax(counts)
    best_M = int(unique_Ms[best_M_idx])
    
    # Extract the k values corresponding to best_M
    best_ks = ks[inverse == best_M_idx]
    
    # Convert to Python list of ints
    k_list = best_ks.tolist()
    
    return {'M': best_M, 'k': k_list}
