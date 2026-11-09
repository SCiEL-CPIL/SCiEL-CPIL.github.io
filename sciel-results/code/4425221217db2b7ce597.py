import numpy as np
import itertools
import random

def construct(q, n, radius):
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Compute coverage matrix
    # dist[i, j] = Hamming distance between all_words[i] and all_words[j]
    # Using broadcasting
    # all_words[:, None, :] shape (N, 1, n)
    # all_words[None, :, :] shape (1, N, n)
    # Comparison gives (N, N, n) boolean array
    
    # To save memory, we can do it in chunks if N is large, but N=2187 is small.
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dist = diff.sum(axis=2)
    cov_matrix = dist <= radius
    
    best_code = None
    best_len = total_words + 1
    
    # Run randomized greedy
    num_runs = 50
    # Use a different seed for each run? Or just randomize tie-breaking.
    
    for run in range(num_runs):
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        
        while uncovered.any():
            # Gain for each candidate
            # new_cov = cov_matrix & uncovered
            # gains = new_cov.sum(axis=1)
            
            # Optimization: 
            # We only care about candidates that cover at least one uncovered word.
            # But computing gains for all is fine.
            
            new_cov = cov_matrix & uncovered
            gains = new_cov.sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Find all candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            chosen = candidates[random.randint(0, len(candidates) - 1)]
            
            selected.append(chosen)
            # Update uncovered
            uncovered = uncovered & ~cov_matrix[chosen]
            
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected.copy()
            
    # Convert to list of lists
    if best_code is None:
        return [list(w) for w in all_words]
        
    return [list(all_words[i]) for i in best_code]
