<<<<<<< SEARCH
    # ---------- Greedy Set-Cover Construction ----------
    # Start with an empty code and greedily add codewords that cover the most
    # currently uncovered words.
    
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    
    while uncovered_mask.any() and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        best_c = -1
        best_count = 0
        
        # Vectorized approach: For each candidate c, count how many of its neighbors
        # are currently uncovered.
        for c in range(N):
            if best_count == N:  # Can't do better
                break
            # Count uncovered neighbors
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
        
        if best_c == -1 or best_count == 0:
            break
        
        # Add best_c to code
        code.append(best_c)
        # Update coverage and uncovered_mask
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False
=======
    # ---------- Greedy Set-Cover Construction ----------
    # Start with an empty code and greedily add codewords that cover the most
    # currently uncovered words.
    
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    
    # Precompute neighbor index arrays for vectorized operations
    # neighbors_list is already a list of numpy arrays
    
    while uncovered_mask.any() and time.time() < deadline:
        # Find the codeword that covers the most uncovered words
        # Use vectorized approach: compute gain for all candidates at once
        # gain[c] = number of uncovered words in ball(c)
        
        # For each candidate c, gain = sum of uncovered_mask over neighbors_list[c]
        # We can compute this by: for each uncovered word w, add 1 to all c that cover w
        # But that's expensive. Instead, use the fact that neighbors_list is precomputed.
        
        # Efficient approach: iterate through uncovered words and accumulate gains
        gain = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            break
        
        # For each uncovered word, find which codewords cover it
        # A codeword c covers word w if w is in neighbors_list[c]
        # Equivalent: c is in neighbors_list[w] (by symmetry of Hamming distance)
        
        for w in uncovered_indices:
            # All codewords c such that w is in ball(c) are exactly neighbors_list[w]
            gain[neighbors_list[w]] += 1
        
        best_c = int(np.argmax(gain))
        best_count = gain[best_c]
        
        if best_count == 0:
            break
        
        # Add best_c to code
        code.append(best_c)
        # Update coverage and uncovered_mask
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False
>>>>>>> REPLACE

