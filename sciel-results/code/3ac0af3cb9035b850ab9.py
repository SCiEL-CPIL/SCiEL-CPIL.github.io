import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_phi_sieve_numpy(n_max):
        """Compute phi(1)..phi(n_max) using a segmented-style sieve with NumPy."""
        phi_arr = np.arange(1, n_max + 1, dtype=np.int64)
        # For each prime p, multiply phi by (p-1)/p for all multiples of p
        is_prime = np.ones(n_max + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        
        for p in range(2, n_max + 1):
            if is_prime[p]:
                # Mark multiples as non-prime
                is_prime[p*p::p] = False
                # Update phi for all multiples of p
                multiples = np.arange(p, n_max + 1, p)
                phi_arr[multiples - 1] = phi_arr[multiples - 1] // p * (p - 1)
        
        return phi_arr
    
    def find_longest_distinct_run(phi_arr, max_start, max_len=200):
        """Find the longest run of consecutive integers with distinct phi values."""
        n = len(phi_arr)
        best_k = 1
        best_s = 1
        
        # Use a sliding window with a count array to detect duplicates
        # We need to track counts of phi values in the current window
        # phi values can be up to n_max, so we use a dict or array
        
        # For efficiency, use a dictionary-based approach for the sliding window
        # Actually, since we're looking for runs up to 200, we can use a set-based approach
        # but with a more efficient method: for each starting position, extend the window
        # until we find a duplicate or reach the max length.
        
        # Better approach: use a frequency array if phi values are bounded
        # phi(n) <= n, so phi values are in [1, n_max]
        # Use a list to track counts
        
        max_val = int(phi_arr.max()) if len(phi_arr) > 0 else 1
        counts = np.zeros(max_val + 1, dtype=np.int32)
        
        # Sliding window: maintain window [left, right]
        left = 0
        right = 0
        current_len = 0
        
        for right in range(n):
            val = int(phi_arr[right])
            counts[val] += 1
            current_len += 1
            
            # Shrink from left while there's a duplicate
            while counts[val] > 1:
                left_val = int(phi_arr[left])
                counts[left_val] -= 1
                left += 1
                current_len -= 1
            
            # Update best
            if current_len > best_k:
                best_k = current_len
                best_s = left + 1  # 1-indexed
        
        return best_k, best_s
    
    # Start with a reasonable block size
    # The seed program found runs up to ~18 in range(1, 20001)
    # We need to search larger ranges to find longer runs
    
    # Strategy: compute phi for a large block, find the best run
    # Then if time permits, try other blocks
    
    # Let's start with a block of size 10^6
    block_size = 10**6
    
    if time_remaining() < 5.0:
        # Fallback: just return a small valid result
        return {'start': 1, 'length': 1}
    
    # Compute phi for the first block
    phi_arr = compute_phi_sieve_numpy(block_size)
    
    best_k, best_s = find_longest_distinct_run(phi_arr, block_size)
    
    # If time permits, try additional blocks
    # The key insight: we need to search over different starting positions
    # Since phi values repeat, we need to check many different ranges
    
    # Try to extend the search to larger ranges
    # But first, let's see if we can find longer runs in the current block
    
    # If the best run is still small, try a larger block
    if time_remaining() > 30.0 and best_k < 50:
        block_size_2 = 5 * 10**6
        if time_remaining() > 60.0:
            phi_arr = compute_phi_sieve_numpy(block_size_2)
            k2, s2 = find_longest_distinct_run(phi_arr, block_size_2)
            if k2 > best_k:
                best_k, best_s = k2, s2
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
