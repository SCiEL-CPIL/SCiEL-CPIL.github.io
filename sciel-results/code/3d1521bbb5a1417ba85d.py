import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline for safety
    
    N = 10_000_000
    target_ratio = 1.5
    
    # 1. Compute sigma(n) for n in 1..N using vectorized sieve
    # sigma is initialized to 1 (divisor 1)
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for sigma: for each i, add i to all multiples of i
    for i in range(2, int(np.sqrt(N)) + 1):
        if sigma[i] == 1:  # i is prime
            p = i
            p_power = p
            # For prime powers p^k, sigma(p^k) = 1 + p + ... + p^k
            # We can update sigma for multiples of p, p^2, etc.
            # Actually, a simpler approach for general sieve:
            # sigma(n) is multiplicative.
            # Let's stick to the additive sieve which is straightforward:
            # sigma[n] += i for all n % i == 0
            # But that's O(N log N). For N=10^7, it's fast in NumPy.
            pass
            
    # The additive sieve is:
    # for i in 1..N:
    #   for j in i, 2i, 3i, ... <= N:
    #     sigma[j] += i
    # In NumPy:
    # for i in range(1, N+1):
    #   sigma[i::i] += i
    # This loop runs 10^7 times. In Python, this loop overhead might be significant.
    # Let's check performance. 10^7 iterations in Python is ~0.5-1s if simple, but slicing is fast.
    # However, doing this in a Python loop for 10^7 steps is risky for 90s.
    # Alternative: Use a more optimized sieve.
    
    # Let's try the standard multiplicative sieve logic with arrays.
    # It's complex to vectorize perfectly.
    # Let's stick to the additive sieve but optimize the loop.
    # Actually, 10^7 iterations in Python is too slow if each does a slice operation that allocates.
    # sigma[i::i] creates a view, addition is in-place? No, `+=` on a slice might copy or be slow.
    # `sigma[i::i] = sigma[i::i] + i` is safer but allocates.
    
    # Better approach for Sigma:
    # Use the fact that sigma(n) = sum_{d|n} d.
    # We can iterate i from 1 to N.
    # For N=10^7, a pure Python loop is too slow.
    # NumPy loop:
    # for i in range(1, N + 1):
    #     sigma[i::i] += i
    # This is 10^7 NumPy operations. Each is small.
    # Let's test if this fits in time.
    
    # Actually, a better way is to use a prime factorization sieve to compute sigma multiplicatively.
    # This is O(N log log N) roughly.
    
    # Let's use the additive sieve but with a twist:
    # We can't easily vectorize the outer loop in pure Python/NumPy without writing C.
    # However, 10^7 iterations of `sigma[i::i] += i` in NumPy is often acceptable (~10-20s).
    # Let's try it.
    
    # Reset sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        # Add i to all multiples of i
        sigma[i::i] += i
        
        # Time check inside the loop? No, too frequent.
        # We'll check after the loop.
        
    # Time check
    if time.time() > deadline - 10.0:
        # If sigma calculation took too long, we might not have time for search.
        # But sigma calc is usually the bottleneck.
        pass

    # 2. Bucket indices by sigma value
    # We need a mapping: sigma_val -> list of indices
    # Using a dictionary of lists.
    # To save memory and time, we can sort indices by sigma value.
    # indices = np.arange(1, N+1)
    # sorted_indices = indices[np.argsort(sigma[1:])]
    # Then we can find contiguous segments where sigma is the same.
    
    idx = np.arange(1, N + 1)
    # Sort indices by their sigma value
    # This is an O(N log N) operation.
    sorted_idx = idx[np.argsort(sigma[1:], kind='mergesort')]
    sorted_sigma = sigma[1:][np.argsort(sigma[1:], kind='mergesort')]
    
    # Find boundaries where sigma value changes
    # diff = np.diff(sorted_sigma)
    # boundaries = np.where(diff != 0)[0] + 1
    # This gives start indices of each group.
    
    # Instead of creating a dict of lists (which is slow in Python),
    # we can iterate through the sorted array and process groups.
    
    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # Iterate through groups
    # We need to find the start and end of each group in sorted_idx
    # Let's use np.unique with return_index?
    # unique_sigmas, start_indices, counts = np.unique(sorted_sigma, return_index=True, return_counts=True)
    
    unique_sigmas, start_indices, counts = np.unique(sorted_sigma, return_index=True, return_counts=True)
    
    # Iterate through each unique sigma group
    for i in range(len(unique_sigmas)):
        if time.time() > deadline - 5.0:
            break
            
        start = start_indices[i]
        count = counts[i]
        
        # If group size is small, pairs are few
        if count < 2:
            continue
            
        # Extract the group of indices
        # group_indices = sorted_idx[start : start + count]
        # These are already sorted because sorted_idx was sorted by sigma, 
        # and within same sigma, the order is stable (mergesort) so it preserves original index order (ascending).
        
        group = sorted_idx[start : start + count]
        
        # For each b in group, find a in group such that a/b ~ 1.5
        # a ~ 1.5 b
        # Since group is sorted, we can use bisect.
        
        # Optimization: If count is very large, bisecting for each element is O(count log count).
        # Total complexity is sum(count_i log count_i) <= N log N.
        
        for j in range(count):
            b = group[j]
            target_a = 1.5 * b
            
            # Find position in group where target_a would be inserted
            # group is a NumPy array, bisect works on sequences.
            # bisect.bisect_left(group, target_a)
            
            pos = bisect.bisect_left(group, target_a)
            
            # Check neighbors: pos-1, pos, pos+1
            # Ensure indices are valid
            
            candidates = []
            if pos - 1 >= 0:
                candidates.append(pos - 1)
            if pos < count:
                candidates.append(pos)
            if pos + 1 < count:
                candidates.append(pos + 1)
                
            for k in candidates:
                a = group[k]
                if a == b:
                    continue
                # Calculate error
                # |a/b - 1.5|
                err = abs(a / b - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = int(a)
                    best_b = int(b)
                    
    return [best_a, best_b]
