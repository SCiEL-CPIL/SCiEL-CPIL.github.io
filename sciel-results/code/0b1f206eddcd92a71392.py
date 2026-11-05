import numpy as np
import itertools

def construct(q, n, radius):
    # We assume radius=1 for this specific task structure, 
    # but the logic below works for small radii if spheres are precomputed correctly.
    # For radius=1, sphere size is 1 + n*(q-1).
    
    N = q ** n
    
    # Precompute spheres of radius 'radius' for all words.
    # For efficiency, we assume radius is small (e.g., 1).
    # If radius > 1, the sphere size grows rapidly. 
    # Given the task is K_3(7,1), radius is 1.
    
    # Generate all words as base-q digits
    # We will store neighbors as an array of indices
    # Shape: (N, sphere_size)
    
    # Calculate sphere size for radius=1
    if radius == 1:
        sphere_size = 1 + n * (q - 1)
        neighbors = np.empty((N, sphere_size), dtype=int)
        
        # Helper to convert index to digits and back
        # Since q and n are small, we can do this on the fly or precompute digits.
        # Precomputing digits for all N words is fast.
        digits = np.zeros((N, n), dtype=int)
        for i in range(N):
            val = i
            for j in range(n):
                digits[i, j] = val % q
                val //= q
        
        # Fill neighbors
        for i in range(N):
            idx = 0
            # The word itself
            neighbors[i, idx] = i
            idx += 1
            
            # Neighbors
            for pos in range(n):
                orig_val = digits[i, pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Calculate new index
                    # index = sum(d_j * q^j)
                    # delta = (new_val - orig_val) * q^pos
                    neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                    idx += 1
        # Note: The above calculation `i + (new_val - orig_val) * (q ** pos)` works 
        # because the base-q representation is little-endian in the index calculation 
        # if we define index = d_0 + d_1*q + ...
        # Let's verify:
        # i = sum(d_k * q^k)
        # new_i = sum(d'_k * q^k) where d'_pos = new_val, others same.
        # new_i = i - orig_val*q^pos + new_val*q^pos = i + (new_val - orig_val)*q^pos.
        # This is correct.
        
    else:
        # Fallback for other radii (not expected here but for safety)
        # This would be much slower. 
        # For the purpose of this task, radius=1 is guaranteed.
        raise NotImplementedError("Radius > 1 not optimized in this greedy strategy")

    covered = np.zeros(N, dtype=bool)
    result_indices = []
    
    # Greedy set cover
    while covered.sum() < N:
        # Calculate gain for each candidate
        # covered[neighbors] gives a (N, sphere_size) boolean array
        # ~covered[neighbors] gives True for uncovered neighbors
        # .sum(axis=1) sums over the sphere for each candidate
        gains = (~covered[neighbors]).sum(axis=1)
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            # Should not happen if the space is fully coverable, 
            # but breaks infinite loop if something is wrong
            break
            
        result_indices.append(best_idx)
        covered[neighbors[best_idx]] = True
        
    # Post-processing: Remove redundant codewords
    # If a codeword's entire sphere is covered by other codewords, we can remove it.
    # We iterate and check.
    # To make this efficient, we can maintain the coverage count or just re-check.
    # Since N is small (2187) and result size is ~150-190, checking each one
    # by subtracting its contribution and seeing if any drop to 0 is feasible.
    
    # Let's use a simpler approach:
    # Keep the list of indices.
    # For each index in the list, check if removing it leaves any point uncovered.
    # We can do this by creating a coverage count array.
    
    coverage_counts = np.zeros(N, dtype=int)
    for idx in result_indices:
        coverage_counts[neighbors[idx]] += 1
        
    # Now, try to remove codewords.
    # We iterate through the list. If we remove one, we decrement counts.
    # If any count becomes 0, the removal is invalid (unless we re-add or accept gap, but we can't).
    # So we only remove if min(coverage_counts[neighbors[idx]]) >= 2 before removal?
    # No, we need to check if *all* points in its sphere have count > 1.
    
    # Efficient check:
    # For a candidate idx, if np.min(coverage_counts[neighbors[idx]]) > 1, then it is safe to remove.
    
    new_result_indices = []
    for idx in result_indices:
        # Check if safe to remove
        if np.min(coverage_counts[neighbors[idx]]) > 1:
            # Safe to remove
            coverage_counts[neighbors[idx]] -= 1
        else:
            # Not safe, keep it
            new_result_indices.append(idx)
            
    result_indices = new_result_indices
    
    # We can repeat this process or just do one pass. One pass is usually good.
    # A second pass might catch more.
    
    # Let's do a second pass just in case
    coverage_counts = np.zeros(N, dtype=int)
    for idx in result_indices:
        coverage_counts[neighbors[idx]] += 1
        
    new_result_indices = []
    for idx in result_indices:
        if np.min(coverage_counts[neighbors[idx]]) > 1:
            coverage_counts[neighbors[idx]] -= 1
        else:
            new_result_indices.append(idx)
            
    result_indices = new_result_indices

    # Convert indices back to lists of integers
    # Re-use the digits array if we had it, or recompute
    # Recomputing is fast for small N
    result_words = []
    for idx in result_indices:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
