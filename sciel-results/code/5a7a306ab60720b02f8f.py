import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute neighbors and mask
    # mask[i, j] is True if j is in the ball of radius 'radius' around i
    # For radius=1, q=3, n=8, this is symmetric.
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
            
    # Generate all words
    words = [to_word(i) for i in range(total_words)]
    
    # Build mask
    # mask is a boolean array of shape (total_words, total_words)
    # For memory efficiency, we can use uint8 or bool. 
    # 6561 * 6561 * 1 byte = ~43 MB. This is acceptable.
    mask = np.zeros((total_words, total_words), dtype=bool)
    
    # Precompute balls
    balls = []
    for i in range(total_words):
        word = words[i]
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        balls.append(ball)
        mask[i, ball] = True
        
    # 1. Greedy Set Cover Initialization
    uncovered_mask = np.ones(total_words, dtype=np.int8)
    code = set()
    
    # Optimization: Precompute neighbor indices as numpy arrays for fast sum
    # We can use mask directly for counting uncovered neighbors
    # count = sum(uncovered_mask[mask[c, :]])
    
    while np.any(uncovered_mask):
        best_c = -1
        best_count = -1
        
        # Vectorized find best candidate
        # For each c, count uncovered neighbors
        # This is a matrix multiplication or sum
        # uncovered_mask is (N,), mask is (N, N)
        # We want sum over columns for each row? No, mask[c, :] is the neighbors of c.
        # So we want sum(uncovered_mask * mask[c, :]) for each c.
        # This is mask @ uncovered_mask
        counts = mask @ uncovered_mask
        # We should ignore c already in code.
        # Since we add to code, we can just mask out those indices.
        # But 'code' is a set, so we need to zero out those rows in counts?
        # Or just check in a loop?
        # Let's zero out rows for c in code.
        for c in code:
            counts[c] = -1
            
        best_c = np.argmax(counts)
        best_count = counts[best_c]
        
        if best_count <= 0:
            break
            
        code.add(best_c)
        uncovered_mask[mask[best_c, :]] = 0
        
    # 2. Local Search: Minimize Code Size
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        cover_count[mask[c, :]] += 1
        
    # Helper to update cover count
    def add_to_cover(c):
        cover_count[mask[c, :]] += 1
        
    def remove_from_cover(c):
        cover_count[mask[c, :]] -= 1
        
    # Iterative Improvement
    max_iterations = 2000
    for _ in range(max_iterations):
        improved = False
        
        # Phase 1: Remove any word that doesn't uniquely cover any point
        # unique_count[c] = sum(1 for w in neighbors[c] if cover_count[w] == 1)
        # If unique_count[c] == 0, we can remove c.
        
        # Vectorized check for redundancy
        # unique_mask = (cover_count == 1)
        # For each c in code, check if any neighbor has cover_count 1.
        # redundant_mask = (mask @ unique_mask) == 0 ? No, mask @ unique_mask gives count of unique neighbors.
        unique_neighbors = mask @ (cover_count == 1)
        
        # Find c in code where unique_neighbors[c] == 0
        to_remove = [c for c in code if unique_neighbors[c] == 0]
        
        if to_remove:
            # Remove all redundant at once? Or one by one?
            # Removing one might make another redundant or non-redundant?
            # Removing a redundant c decreases cover counts.
            # This might make other points unique, or leave them unique.
            # It cannot make a point *less* unique (count < 1 is impossible if covered).
            # It might make a point that was count 2 become count 1 (new unique).
            # So removing redundant codewords is always safe and might enable more removals.
            for c in to_remove:
                code.remove(c)
                remove_from_cover(c)
            improved = True
            continue
        
        # Phase 2: Try Swaps
        # For each c in code, find c' not in code such that c' covers all unique points of c.
        # unique_points_c = [w for w in range(N) if mask[c, w] and cover_count[w] == 1]
        # We need c' such that for all w in unique_points_c, mask[c', w] is True.
        # This is equivalent to: mask[c', unique_points_c].all()
        
        code_list = list(code)
        random.shuffle(code_list)
        
        swapped = False
        for c_out in code_list:
            # Get unique points for c_out
            # unique_mask_c = mask[c_out, :] & (cover_count == 1)
            # unique_points_indices = np.where(unique_mask_c)[0]
            
            unique_mask_c = mask[c_out, :] & (cover_count == 1)
            unique_points_count = np.sum(unique_mask_c)
            
            if unique_points_count == 0:
                # Should have been removed in Phase 1
                continue
                
            # Find c' not in code that covers all these points
            # We need mask[:, unique_points].all(axis=1)
            # But we can't index columns with a boolean mask directly in a way that produces a boolean column vector easily without fancy indexing.
            # Let's get the indices of unique points.
            unique_indices = np.where(unique_mask_c)[0]
            
            if len(unique_indices) == 0:
                continue
                
            # Check all c'
            # candidate_mask = mask[:, unique_indices].all(axis=1)
            # This creates an array of size N.
            # We need to exclude c' in code.
            
            # Optimization: If unique_indices is large, this is fast. If small, it's fast too.
            # mask[:, unique_indices] has shape (N, len(unique_indices))
            # .all(axis=1) checks if all are True.
            
            # We can speed this up by checking a random subset of c' first?
            # Or just check all. N=6561, len(unique) <= 17.
            # 6561 * 17 operations is ~100k. Very fast.
            
            # However, doing this for every c_out in code (500 of them) is 50M ops.
            # 50M simple ops in NumPy is fast (< 1s).
            
            candidates_mask = mask[:, unique_indices].all(axis=1)
            
            # Exclude current code
            for c_in in code:
                candidates_mask[c_in] = False
                
            # Find any valid c_in
            valid_indices = np.where(candidates_mask)[0]
            
            if len(valid_indices) > 0:
                # Pick one randomly or the one that minimizes something else?
                # For now, pick the first one.
                best_c_in = valid_indices[0]
                
                # Perform swap
                code.remove(c_out)
                remove_from_cover(c_out)
                code.add(best_c_in)
                add_to_cover(best_c_in)
                
                swapped = True
                improved = True
                break # Restart loop to recalculate unique counts
        
        if not improved:
            break
            
    # Convert back to list of lists
    result = [to_word(c) for c in code]
    return result
