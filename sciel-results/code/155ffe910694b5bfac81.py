import numpy as np
import time

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    target_ratio = 1.5
    
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s margin before 90s limit

    # 1. Compute sigma(n) for n in [1, N]
    # Using the additive divisor sum sieve: sigma[n] = sum of divisors of n
    # sigma[i::i] += i
    # This is O(N log N) but vectorized in NumPy.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # We can optimize by only iterating i up to N, but the slice length decreases.
    # For i > N/2, sigma[i::i] is just sigma[i] += i.
    # Actually, the loop goes to N.
    # To speed up, we can process the first half with slices, and the second half with a vectorized addition?
    # No, the slice `sigma[i::i]` for i > N/2 has length 1.
    # A Python loop of 10M iterations is ~0.5s-1s.
    # The vectorized ops inside are fast.
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() - start_time > timeout:
        # Fallback to a known good pair if computation is too slow (unlikely)
        # (6, 11) sigma(6)=12, sigma(11)=12. Ratio 11/6 = 1.833. Err 0.333.
        return [11, 6]

    # 2. Group indices by sigma value
    # We need to find pairs (a, b) with sigma[a] == sigma[b].
    # Sort indices 1..N by their sigma value.
    
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    # argsort is stable by default in recent numpy, but we specify it.
    # We sort indices based on sigma[indices]
    order = np.argsort(sigma[indices], kind='stable')
    sorted_indices = indices[order]
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() - start_time > timeout:
        return [11, 6]

    # 3. Find best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We iterate through the sorted array to find blocks of equal sigma values.
    # Instead of creating a list of blocks, we can iterate and track the current block.
    
    n_items = len(sorted_indices)
    i = 0
    while i < n_items:
        if time.time() - start_time > timeout:
            break
            
        current_sigma = sorted_sigmas[i]
        j = i + 1
        
        # Find the end of the block with current_sigma
        # Using searchsorted to find the first index where value > current_sigma
        # Since sorted_sigmas is sorted, we can use searchsorted.
        # But we are already in a loop.
        # It's faster to just increment j until change, or use searchsorted on the whole array?
        # Searchsorted on the whole array for each block start is O(log N) per block.
        # Number of blocks is ~ N / log N? Or less.
        # Let's use searchsorted to find the end of the current block efficiently.
        
        # Find the first index > i where sigma > current_sigma
        # Since sorted_sigmas is sorted, all elements from i to k-1 have value current_sigma.
        # k is the first index where sorted_sigmas[k] != current_sigma.
        
        # We can use np.searchsorted(sorted_sigmas, current_sigma, side='right')
        # But this searches the whole array every time? No, we can restrict the range?
        # np.searchsorted doesn't take a range easily without slicing (which copies).
        
        # Alternative: Just loop j?
        # If blocks are small, looping j is fast.
        # If blocks are large, there are few of them.
        # The total number of increments of j across the whole process is N.
        # So the total time for finding block boundaries is O(N).
        
        while j < n_items and sorted_sigmas[j] == current_sigma:
            j += 1
            
        # Block is sorted_indices[i:j]
        block_size = j - i
        
        if block_size >= 2:
            # Get the block of numbers
            block = sorted_indices[i:j]
            
            # We want to find a, b in block such that |a/b - 1.5| is minimized.
            # This is equivalent to finding a closest to 1.5 * b.
            # Since block is sorted, we can iterate b and binary search for 1.5*b.
            
            # Optimization: If the block is large, iterating all b might be slow.
            # But sum of block sizes is N.
            # So total iterations of the inner loop is N.
            # Each iteration does a binary search (log block_size).
            # Total complexity O(N log N).
            
            # To speed up binary search, we can use np.searchsorted.
            # But calling np.searchsorted 10M times is slow due to function call overhead.
            # We can convert the block to a Python list? No, that's slow too.
            
            # Let's try a two-pointer approach if possible?
            # We want a/b ~ 1.5 => a ~ 1.5 b.
            # Let a and b be pointers in the block.
            # Initialize a_idx = 0, b_idx = 0.
            # While a_idx < len(block) and b_idx < len(block):
            #   a = block[a_idx], b = block[b_idx]
            #   if a == b: a_idx += 1 (or b_idx += 1)
            #   ratio = a / b
            #   if ratio < 1.5: a_idx += 1 (need larger a)
            #   else: b_idx += 1 (need larger b)
            #   update best error.
            
            # This two-pointer approach is O(block_size) per block!
            # Total complexity O(N).
            # This is much faster.
            
            a_idx = 0
            b_idx = 0
            len_block = block_size
            
            while a_idx < len_block and b_idx < len_block:
                a = block[a_idx]
                b = block[b_idx]
                
                if a == b:
                    a_idx += 1
                    continue
                    
                # Calculate error
                # We want to minimize |a/b - 1.5|
                # Equivalent to |2a - 3b| / (2b)
                # Since b is positive, minimizing |2a - 3b| / b is roughly minimizing |2a - 3b|.
                # But we must be careful with the division by b.
                # However, for a fixed block, b varies.
                # Let's just compute the float error.
                
                ratio = a / b
                err = abs(ratio - 1.5)
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
                # Move pointers
                # If a < 1.5 b, we need to increase a (or decrease b, but b is fixed in this step? No, both move)
                # If a/b < 1.5, then 2a < 3b. We need to increase a to get closer to 1.5.
                # If a/b > 1.5, then 2a > 3b. We need to increase b to get closer to 1.5.
                
                if a < 1.5 * b:
                    a_idx += 1
                else:
                    b_idx += 1
                    
        i = j

    # Ensure return types are Python int
    if best_error == float('inf'):
        # Fallback
        return [11, 6]
        
    return [int(best_a), int(best_b)]
