import numpy as np
import time

def construct(q, n, radius):
    # Parameters for the task
    # q=3, n=6, radius=1
    total_words = q ** n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    # Generate all possible words as indices 0 to q^n - 1
    # Convert index to base-q representation
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage masks for all possible codewords
    # masks[c] is a boolean array of length total_words, True if word c covers word w
    masks = np.zeros((total_words, total_words), dtype=bool)
    
    for c_idx in range(total_words):
        c_word = idx_to_word(c_idx)
        masks[c_idx, c_idx] = True # Distance 0
        
        # Distance 1 neighbors
        for pos in range(n):
            orig_val = c_word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    neighbor = c_word.copy()
                    neighbor[pos] = new_val
                    n_idx = word_to_idx(neighbor)
                    masks[c_idx, n_idx] = True

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected_codewords = []
    selected_indices = set()
    
    # Sort candidates by potential coverage? Not necessary, greedy handles it.
    # We will iterate until uncovered is empty or time is up.
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find the candidate that covers the most uncovered words
        # Calculate coverage for all candidates
        # cov_counts[i] = number of True in masks[i] where uncovered is True
        # Using matrix multiplication or logical sum
        # masks shape: (total_words, total_words)
        # uncovered shape: (total_words,)
        # We want sum over axis=1 of (masks & uncovered[:, None])
        
        # Efficient way: 
        # coverage_counts = np.sum(masks & uncovered[None, :], axis=1)
        # But we should ignore already selected ones to avoid re-selection (though they will have 0 new coverage if we update uncovered correctly, 
        # but actually selected ones cover parts of uncovered? No, we update uncovered by removing what the selected covers.
        # So selected ones will have 0 coverage on current uncovered? 
        # Let's check: if we pick c, we set uncovered[masks[c]] = False.
        # Next iteration, for candidate c, masks[c] & uncovered will be 0 at the spots we just cleared.
        # It might still cover other spots if it wasn't fully covered? No, masks[c] is fixed.
        # If we pick c, all spots in masks[c] become False in uncovered.
        # So next time, sum(masks[c] & uncovered) will be 0.
        # So we don't strictly need to filter selected, but it's cleaner.
        
        coverage_counts = np.sum(masks & uncovered[None, :], axis=1)
        
        # Find max
        max_idx = np.argmax(coverage_counts)
        max_cov = coverage_counts[max_idx]
        
        if max_cov == 0:
            # No candidate covers any new word? This shouldn't happen if uncovered is not empty
            # unless there's a bug. But if radius < 1 or something weird.
            # For radius 1, every word covers itself. So if any uncovered word exists,
            # the codeword equal to that word covers it. So max_cov >= 1.
            break
            
        selected_codewords.append(idx_to_word(max_idx))
        selected_indices.add(max_idx)
        
        # Update uncovered
        uncovered[masks[max_idx]] = False

    return selected_codewords
