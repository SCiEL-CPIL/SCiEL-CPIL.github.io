import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code using a Greedy algorithm followed by a Local Search (removal) phase.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words in the space
    N = q ** n
    
    # Pre-compute the "ball" of radius 'radius' for each word.
    # For n=6, q=3, radius=1, ball size is 1 + 6*2 = 13.
    # We represent words as integers 0 to N-1.
    # We need a function to convert integer to vector and vice versa, or just work with vectors.
    # Given N=729 is small, we can precompute balls as lists of integers.
    
    # Helper to convert index to vector
    def idx_to_vec(idx):
        vec = [0] * n
        for i in range(n):
            vec[i] = idx % q
            idx //= q
        return vec
    
    # Helper to convert vector to index
    def vec_to_idx(vec):
        idx = 0
        for i in range(n):
            idx = idx * q + vec[i]
        return idx

    # Precompute balls
    # balls[i] is a list of indices j such that dist(i, j) <= radius
    balls = [[] for _ in range(N)]
    
    # Generate all vectors
    vectors = [idx_to_vec(i) for i in range(N)]
    
    for i in range(N):
        v_i = vectors[i]
        ball_i = [i]
        if radius >= 1:
            # For each position, change value to other possibilities
            for pos in range(n):
                orig_val = v_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    v_new = v_i.copy()
                    v_new[pos] = new_val
                    idx_new = vec_to_idx(v_new)
                    ball_i.append(idx_new)
        balls[i] = ball_i

    # Coverage count array
    coverage = np.zeros(N, dtype=np.int32)
    
    # Code as a list of indices
    code = []
    
    # --- Phase 1: Greedy Construction ---
    # We want to cover all words (coverage > 0 for all)
    # Start with empty code.
    
    while True:
        if time.time() > deadline:
            break
        
        # Find uncovered words
        uncovered_mask = (coverage == 0)
        if not np.any(uncovered_mask):
            break
            
        # For each candidate in the space, count how many uncovered words it covers
        # We only need to consider candidates that cover at least one uncovered word?
        # Actually, any candidate is a valid addition, but we want the best one.
        
        best_candidate = -1
        max_covers = 0
        
        # Optimization: Only consider candidates that are in the space.
        # We can iterate through all N candidates.
        for c in range(N):
            # If candidate c doesn't cover any uncovered word, skip (optimization)
            # But checking this is as expensive as counting.
            
            ball_c = balls[c]
            # Count how many elements in ball_c have coverage 0
            # Using numpy slicing is fast
            count = np.sum(coverage[ball_c] == 0)
            
            if count > max_covers:
                max_covers = count
                best_candidate = c
                
        if best_candidate == -1:
            break # Should not happen if there are uncovered words
            
        # Add best_candidate to code
        code.append(best_candidate)
        
        # Update coverage
        for idx in balls[best_candidate]:
            coverage[idx] += 1
            
    # --- Phase 2: Local Search (Removal) ---
    # Try to remove words from the code if they are redundant.
    # A word is redundant if removing it does not leave any word with coverage 0.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards to allow removal
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            
            # Temporarily remove c
            for idx in balls[c]:
                coverage[idx] -= 1
                
            # Check if any word is now uncovered
            if np.min(coverage) == 0:
                # Restore c
                for idx in balls[c]:
                    coverage[idx] += 1
            else:
                # Keep c removed
                code.pop(i)
                improved = True
                # Note: We don't break the loop, we continue checking other words
                # But since we modified the code list, indices might shift?
                # We are iterating backwards, so popping from the end is safe.
                # If we pop index i, indices < i are unaffected.
                
    return [vectors[c] for c in code]

