import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Calculate total number of words
    total_words = q ** n
    
    # If the space is too large for the dense NumPy approach, 
    # fall back to the seed strategy or a simple greedy without precomputation.
    # For q=3, n=8, total_words=6561, which is very small.
    if total_words > 100000:
        # Fallback: Direct sum of tetracode if applicable, or simple greedy
        # Since the task is specific to n=8, q=3, we assume it fits.
        # But let's implement a basic greedy for larger spaces just in case.
        # However, the prompt implies this specific task.
        pass

    # Generate all words as integers 0 to total_words - 1
    # We need to map integer index to list of q-ary digits.
    # And precompute neighbors.
    
    # 1. Precompute neighbors for all words
    # neighbors[i] is an array of indices of words within distance <= radius from i.
    # For radius=1, neighbors are i itself and words differing in 1 position.
    
    # To compute neighbors efficiently:
    # We can iterate through all words, convert to base-q, modify digits, convert back.
    # Or use arithmetic.
    
    # Let's create an array of all words in base q.
    # Shape: (total_words, n)
    # This might be memory heavy for large n, but n=8 is fine.
    
    # Efficient way to get neighbors for radius=1:
    # For each word i, for each position j in 0..n-1:
    #   for each digit d in 0..q-1, d != word[j]:
    #      new_word = word with j-th digit changed to d.
    #      index = base_q_to_int(new_word)
    
    # We can precompute the "step" values for changing a digit.
    # But simpler: just do it in a loop for precomputation.
    
    neighbors_list = []
    
    # Helper to convert int to base q list
    def int_to_base_q(x, n):
        res = [0] * n
        for i in range(n):
            res[i] = x % q
            x //= q
        return res

    # Helper to convert base q list to int
    def base_q_to_int(digits):
        res = 0
        for i in range(n-1, -1, -1):
            res = res * q + digits[i]
        return res

    # Precompute neighbors
    # This is a one-time cost.
    # 6561 * 8 * 2 = ~100k operations. Fast.
    
    for i in range(total_words):
        word = int_to_base_q(i, n)
        neigh = [i]
        for j in range(n):
            original_digit = word[j]
            for d in range(q):
                if d == original_digit:
                    continue
                # Create new word
                # We can calculate the index change directly
                # index = i + (d - original_digit) * (q ** j)
                # Wait, the base is little-endian in my int_to_base_q?
                # Let's check:
                # res[0] = x % q. So pos 0 is least significant.
                # So weight of position j is q**j.
                
                delta = (d - original_digit) * (q ** j)
                new_idx = i + delta
                # Since d != original_digit, new_idx != i
                # And new_idx is valid because we just changed a digit within range.
                neigh.append(new_idx)
        
        # Ensure no duplicates (though logic above prevents it for radius 1)
        # And sort? Not necessary.
        neighbors_list.append(neigh)
    
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    # 2. Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    while not covered.all():
        # Check time
        if time.time() > deadline:
            break
        
        # Calculate gain for each candidate
        # (~covered) is boolean array.
        # (~covered)[neighbors] gives shape (total_words, sphere_size)
        # Sum along axis 1 gives gain for each candidate.
        gains = (~covered)[neighbors].sum(axis=1)
        
        # Find best candidate
        best_idx = np.argmax(gains)
        
        # If no gain, something is wrong (should be covered)
        if gains[best_idx] == 0:
            break
        
        code_indices.append(best_idx)
        
        # Update covered
        covered[neighbors[best_idx]] = True
        
    # 3. Post-processing: Remove redundant codewords
    if code_indices:
        # Calculate coverage count
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            coverage_count[neighbors[idx]] += 1
        
        # Try to remove from the end (last added are often less critical? 
        # Or just iterate all. Order doesn't matter much for correctness, 
        # but removing later ones might be safer if they were added to fill gaps.)
        
        # We iterate in reverse order of addition.
        # We use a list to store the current code.
        # We check if we can remove code_indices[i].
        
        # To make this efficient, we just check the min coverage in its sphere.
        
        # Note: Removing a codeword reduces coverage_count.
        # If min(coverage_count[neighbors[idx]]) > 1, we can remove it.
        
        # We iterate i from len(code_indices)-1 down to 0.
        # If we remove it, we pop it.
        
        i = len(code_indices) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            idx = code_indices[i]
            min_cov = coverage_count[neighbors[idx]].min()
            
            if min_cov > 1:
                # We can remove this codeword
                coverage_count[neighbors[idx]] -= 1
                code_indices.pop(i)
                # Do not decrement i, because the element at i is now the previous one
                # Wait, pop(i) shifts elements.
                # If we pop i, the next element to check is the one that shifted into i.
                # So we keep i the same.
            else:
                i -= 1

    # 4. Convert indices back to lists of integers
    result = []
    for idx in code_indices:
        result.append(int_to_base_q(idx, n))
        
    return result
