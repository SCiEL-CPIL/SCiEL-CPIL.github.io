import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe (90s wall clock limit)
    
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N using NumPy sieve
    # sigma(n) = sum of divisors
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for i in 1..N, add i to all multiples of i
    # This is O(N log N) ~ 1.6 * 10^8 operations
    # NumPy slice assignment is fast but the Python loop over i is the bottleneck
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check deadline after sieve
    if time.time() > deadline:
        # Fallback to a known good pair
        # sigma(11) = 12, sigma(6) = 12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333
        # sigma(122) = 186, sigma(80) = 186. |122/80 - 1.5| = |1.525 - 1.5| = 0.025
        return [122, 80]
    
    # Step 2: Group indices by their sigma value
    # We need to find pairs (a, b) with sigma(a) = sigma(b) and minimize |a/b - 1.5|
    # a/b ~ 1.5 => a ~ 1.5 b => 2a ~ 3b => 2a - 3b ~ 0
    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
    # To minimize this, for a given b, we want a such that 2a is close to 3b, i.e., a is close to 1.5b.
    # Since a and b are in the same sigma group, we can sort the group and for each b, 
    # binary search for a close to 1.5b.
    
    # To avoid sorting 10^7 elements individually or using too much memory for lists,
    # we can sort the indices by sigma value.
    
    # indices = 1..N
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    # Sort indices by sigma value
    # We use argsort on sigma[1:N+1]
    # This gives us the permutation of indices that sorts the sigma values
    sorted_indices = indices[np.argsort(sigma[1:N+1], kind='mergesort')]
    sorted_sigmas = sigma[1:N+1][np.argsort(sigma[1:N+1], kind='mergesort')]
    
    # Now iterate through the sorted arrays to find groups of equal sigma
    # For each group, if size > 1, we sort the numbers in that group (by value)
    # and then for each b in the group, find a in the group closest to 1.5b
    
    best_error = float('inf')
    best_a = 122
    best_b = 80
    
    # To speed up, we can identify group boundaries
    # A group starts at i if sorted_sigmas[i] != sorted_sigmas[i-1]
    # We only care about groups with size >= 2
    
    # Pre-allocate a buffer for the current group?
    # Most groups are size 1.
    
    current_sigma = -1
    group_start = 0
    N_items = N
    
    # We can vectorize the group finding partially?
    # Let's just loop through.
    # Optimization: Only process groups if time allows?
    # The loop over N_items is 10^7 iterations. This might be slow in Python.
    # Let's try to vectorize the group boundary detection.
    
    # Create a boolean mask for group starts
    # start_mask[i] = True if sorted_sigmas[i] != sorted_sigmas[i-1] (or i==0)
    # But we need the ranges.
    
    # Alternative:
    # Use np.unique on sorted_sigmas to get unique values and their counts?
    # np.unique(sorted_sigmas, return_counts=True)
    # This returns unique values and counts.
    # But we also need the indices into sorted_indices.
    # np.unique(..., return_index=True) gives the first occurrence index.
    
    unique_sigmas, first_indices, counts = np.unique(sorted_sigmas, return_index=True, return_counts=True)
    
    # Now iterate over the unique sigmas
    # For each sigma, if count > 1, extract the group from sorted_indices
    # group = sorted_indices[first_indices[i] : first_indices[i] + counts[i]]
    
    # Note: sorted_indices are NOT sorted by value within the group, because argsort was on sigma.
    # So we must sort the group by value for binary search.
    
    for i in range(len(unique_sigmas)):
        if time.time() > deadline:
            break
            
        if counts[i] < 2:
            continue
            
        # Extract group
        start_idx = first_indices[i]
        end_idx = start_idx + counts[i]
        group = sorted_indices[start_idx:end_idx].copy()
        
        # Sort group by value
        group.sort()
        
        # For each b in group, find a in group closest to 1.5b
        # Vectorized search:
        # targets = 1.5 * group
        # pos = np.searchsorted(group, targets, side='left')
        
        # We need to check pos and pos-1 for each b
        # This can be vectorized.
        
        targets = 1.5 * group.astype(np.float64)
        pos = np.searchsorted(group, targets, side='left')
        
        # Candidate a's:
        # a1 = group[pos] (if pos < len(group))
        # a2 = group[pos-1] (if pos > 0)
        
        # We need to handle boundaries carefully.
        # Let's create arrays for candidate a's and their corresponding b's.
        
        # Mask for valid pos (pos < len(group))
        valid_pos = pos < len(group)
        # Mask for valid pos-1 (pos > 0)
        valid_pos_minus_1 = pos > 0
        
        # Candidate 1: a = group[pos], b = group
        # Only for valid_pos
        if np.any(valid_pos):
            a_cand = group[pos[valid_pos]]
            b_cand = group[valid_pos]
            # Calculate error: |a/b - 1.5|
            # We want to minimize |2a - 3b| / (2b)
            # Let's just compute float error for now, it's safer for comparison.
            # But we must ensure a != b?
            # If a == b, error is 0.5.
            # If we find error < 0.5, it's better.
            # The problem says "Minimize |a/b - 3/2|".
            # It does not explicitly forbid a=b, but usually "pair" implies distinct?
            # "Approximate a ratio inside a divisor-sum fiber"
            # If a=b, ratio is 1. |1 - 1.5| = 0.5.
            # If we find a pair with error < 0.5, we are better.
            # Let's assume a != b is required or at least preferred if it leads to lower error.
            # Actually, if a=b, error is 0.5.
            # If we find a pair with error 0.025, that's better.
            # So we just minimize the error.
            
            errors = np.abs(a_cand / b_cand - 1.5)
            
            # Find min error in this batch
            min_idx = np.argmin(errors)
            min_err = errors[min_idx]
            
            if min_err < best_error:
                best_error = min_err
                best_a = int(a_cand[min_idx])
                best_b = int(b_cand[min_idx])
                
        # Candidate 2: a = group[pos-1], b = group
        if np.any(valid_pos_minus_1):
            a_cand = group[pos[valid_pos_minus_1] - 1]
            b_cand = group[valid_pos_minus_1]
            
            errors = np.abs(a_cand / b_cand - 1.5)
            
            min_idx = np.argmin(errors)
            min_err = errors[min_idx]
            
            if min_err < best_error:
                best_error = min_err
                best_a = int(a_cand[min_idx])
                best_b = int(b_cand[min_idx])
                
    # Return best pair
    # Ensure types are Python int
    return [int(best_a), int(best_b)]
