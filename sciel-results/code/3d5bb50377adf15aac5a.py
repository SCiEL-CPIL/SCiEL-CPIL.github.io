import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    # Total words: q**n = 729
    
    total_words = q ** n
    
    # Map word tuple to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute spheres as a matrix for fast NumPy operations
    # sphere_matrix[i, j] is 1 if word j is in the sphere of word i
    sphere_matrix = np.zeros((total_words, total_words), dtype=np.uint8)
    
    for i, w in enumerate(all_words):
        # Self
        sphere_matrix[i, i] = 1
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                j = word_to_idx(new_w)
                sphere_matrix[i, j] = 1

    # Initial best solution: the direct product construction from the seed
    # This gives 81 codewords.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(h + list(tail))
            
    best_codewords = initial_codewords
    best_size = len(best_codewords)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # NumPy Greedy Set Cover with Randomization
    # We run multiple restarts.
    
    for restart in range(2000): # Run up to 2000 restarts
        if time.time() > deadline:
            break
            
        uncovered = np.ones(total_words, dtype=np.uint8)
        current_codewords = []
        
        while True:
            # Calculate gains: how many uncovered words each candidate covers
            gains = sphere_matrix @ uncovered
            
            if gains.max() == 0:
                break # All covered
            
            # Randomized choice among top-k candidates to diversify
            max_gain = gains.max()
            candidates = np.where(gains == max_gain)[0]
            
            # Pick randomly from max gain candidates
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            current_codewords.append(chosen_idx)
            
            # Update uncovered
            uncovered = uncovered & (~sphere_matrix[chosen_idx])
            
            # Optimization: if uncovered is all 0, break early
            if uncovered.sum() == 0:
                break

        # Local Search: Prune redundant codewords from current_codewords
        # current_codewords is a list of indices
        if len(current_codewords) < best_size:
            # Try to prune
            # We need to maintain the list of indices
            cw_indices = current_codewords[:]
            
            # Calculate initial counts
            if len(cw_indices) > 0:
                counts = sphere_matrix[cw_indices].sum(axis=0)
            else:
                counts = np.zeros(total_words, dtype=np.uint8)
            
            improved = True
            while improved:
                improved = False
                # Iterate through current codewords
                # We need to check if removing cw[i] leaves all words covered
                # i.e., counts[sphere_matrix[cw[i]]] > 1 for all j in sphere_matrix[cw[i]]
                for i in range(len(cw_indices)):
                    cw = cw_indices[i]
                    # Get the sphere of this codeword
                    # We need to check counts for these positions
                    # mask = sphere_matrix[cw]
                    # if counts[mask].min() > 1:
                    
                    # Efficient check:
                    # min_count_in_sphere = counts[sphere_matrix[cw]].min()
                    # If min_count_in_sphere > 1, we can remove.
                    
                    # Note: sphere_matrix[cw] is a row of 0s and 1s.
                    # counts[sphere_matrix[cw]] will select the counts where mask is 1?
                    # No, numpy boolean indexing works on 1D arrays.
                    # sphere_matrix[cw] is 1D boolean/uint8.
                    # So counts[sphere_matrix[cw] == 1] gives the counts.
                    
                    sphere_mask = sphere_matrix[cw]
                    # Get counts for covered words
                    covered_counts = counts[sphere_mask]
                    
                    if covered_counts.min() > 1:
                        # Remove this codeword
                        # Update counts
                        counts -= sphere_mask
                        # Remove from list
                        cw_indices.pop(i)
                        improved = True
                        break # Restart loop as indices changed
            
            if len(cw_indices) < best_size:
                best_size = len(cw_indices)
                best_codewords = [all_words[i] for i in cw_indices]
                
                if best_size <= 73:
                    # Excellent, but keep looking for 72/71 if time permits
                    pass

    return best_codewords
