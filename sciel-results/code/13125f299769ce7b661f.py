"""Greedy set-cover for ternary covering code K_3(6,1)."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    # Total space size
    total = q ** n
    
    # Precompute all words as a 2D array of shape (total, n)
    # Each row is one word in {0..q-1}^n
    all_words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        x = i
        for j in range(n):
            all_words[i, j] = x % q
            x //= q
    
    # Precompute the neighborhood of each word (Hamming distance <= radius)
    # For radius=1, each word covers itself + q*(n-1) neighbors
    # We'll build a coverage matrix implicitly
    
    # Track which words are covered
    covered = np.zeros(total, dtype=bool)
    covered_count = 0
    
    # Codewords selected so far
    codewords = []
    
    # For efficiency, precompute for each position j and each value v,
    # which words have value v at position j
    # words_at_pos_val[j][v] = list of indices where word has value v at position j
    words_at_pos_val = []
    for j in range(n):
        pos_vals = []
        for v in range(q):
            mask = all_words[:, j] == v
            pos_vals.append(np.where(mask)[0])
        words_at_pos_val.append(pos_vals)
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    # The neighborhood of word w is: w itself + all words differing in exactly one position
    
    # To speed up, we can compute the number of newly covered words for each candidate
    # But 729 candidates x 729 space = manageable with vectorization
    
    while covered_count < total:
        if time.time() > deadline:
            break
        
        # Find the best candidate: maximize newly covered words
        # For each candidate c, its neighborhood is:
        #   c itself + {words differing from c in exactly one position}
        
        # Compute coverage gain for each candidate
        # A word u is in the neighborhood of c iff Hamming distance <= radius
        # For radius=1: u = c or u differs from c in exactly one position
        
        best_gain = -1
        best_candidate = -1
        
        # For each candidate, compute how many uncovered words it covers
        # We'll iterate over candidates and compute their neighborhood's intersection with uncovered set
        
        # Optimization: precompute the neighborhood indices for each word
        # But that's 729 * (1 + 6*2) = 729 * 13 = 9477 indices total, very small
        
        # Let's compute neighborhoods on the fly but vectorized
        # For candidate c, neighborhood = {c} union over j of {words with same as c except position j}
        
        # We can compute this as:
        # neighbor_mask = (all_words == c).all(axis=1)  # just c
        # For each j: words that match c in all positions except j
        #   that's: for all k != j, all_words[:,k] == c[k]
        
        # Let's build the neighborhood index list for each candidate
        # Since n=6, q=3, each neighborhood has 1 + 6*2 = 13 words
        
        # Precompute neighborhood indices for all 729 words
        # neigh_indices[i] = list of word indices in neighborhood of word i
        neigh_indices = []
        for i in range(total):
            c = all_words[i]
            # Start with self
            neighbors = {i}
            # For each position j, change to each other value
            for j in range(n):
                for v in range(q):
                    if v == c[j]:
                        continue
                    # Create the modified word
                    mod = c.copy()
                    mod[j] = v
                    # Find its index: convert back to integer
                    idx = 0
                    for k in range(n):
                        idx = idx * q + mod[k]
                    neighbors.add(idx)
            neigh_indices.append(sorted(neighbors))
        
        # Now greedy selection
        while covered_count < total:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_candidate = -1
            
            for cand in range(total):
                # Skip if this candidate is already a codeword (its neighborhood is already covered)
                # Actually, we should skip candidates whose entire neighborhood is already covered
                # But simpler: just compute gain for each
                neigh = neigh_indices[cand]
                # Count uncovered in neighborhood
                gain = 0
                for idx in neigh:
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
                if best_gain == len(neigh):  # max possible
                    break
            
            if best_gain <= 0:
                break
            
            # Add the best candidate
            codewords.append(all_words[best_candidate].tolist())
            for idx in neigh_indices[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            
            # Periodic check
            if time.time() > deadline:
                break
        
        if covered_count < total:
            # If we ran out of time, break
            break
    
    # Verify coverage is complete; if not, fall back to a valid construction
    if covered_count < total:
        # Fallback: use the direct sum construction (always valid)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return codewords
