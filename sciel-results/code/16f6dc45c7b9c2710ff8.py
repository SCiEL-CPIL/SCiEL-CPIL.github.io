import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover matrix using NumPy for speed
    # cover_matrix[i, j] is True if word j is within Hamming distance <= radius of word i
    # Shape: (total_space, total_space) boolean
    # For 2187x2187, this is ~4.7M booleans, which is ~4.7MB. Very manageable.
    
    # We can construct this efficiently.
    # For radius 1, word j is covered by i if dist(i, j) <= 1.
    # dist(i, j) <= 1 means i == j OR they differ in exactly 1 position.
    
    # Let's build the cover matrix.
    # cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    # for i in range(total_space):
    #    cover_matrix[i, i] = True
    #    w = all_words[i]
    #    for pos in range(n):
    #        orig_val = w[pos]
    #        for val in range(q):
    #            if val == orig_val: continue
    #            nw = w[:]
    #            nw[pos] = val
    #            j = word_to_idx(nw)
    #            cover_matrix[i, j] = True
            
    # Vectorized construction for radius 1:
    # We can compute the distance matrix or directly the cover matrix.
    # Since n=7, q=3, we can iterate over all pairs? 2187^2 = 4.7M. 
    # Python loop over 4.7M pairs might be slow (~10-20s).
    # Let's try a smarter way.
    
    # For each position pos, and each value change, we can map indices.
    # Actually, let's just use the loop but optimize it or use numpy tricks.
    # A better way:
    # cover_matrix = np.eye(total_space, dtype=bool)
    # For each position p in 0..n-1:
    #   For each index i:
    #     For each new value v != word[i][p]:
    #       j = index of word with v at p
    #       cover_matrix[i, j] = True
       
    # Let's precompute the "neighbors" lists first as lists of ints, then fill numpy array.
    # Or just fill numpy array directly in a loop.
    
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    
    # Diagonal is True
    np.fill_diagonal(cover_matrix, True)
    
    # For radius 1
    if radius >= 1:
        # Iterate over all words
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Calculate j efficiently
                    # j = i + (val - orig_val) * (q ** pos) ? No, base q arithmetic.
                    # idx = sum(w[k] * q^(n-1-k))
                    # If we change w[pos] to val, the change in index is (val - orig_val) * q^(n-1-pos)
                    # Let's verify index formula.
                    # idx_to_word uses reverse: w[0] is most significant?
                    # idx_to_word: 
                    #   w.append(idx % q); idx //= q
                    #   w.reverse()
                    # So w[0] is the last extracted, which is the most significant digit.
                    # w[k] corresponds to q^(n-1-k).
                    # So changing w[pos] to val changes index by (val - orig_val) * (q ** (n-1-pos)).
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Greedy Set Cover using NumPy
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=bool)
        code = []
        unc_int8 = uncovered.astype(np.int8)
        
        while np.any(uncovered):
            gains = cm_int8.dot(unc_int8)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Tie-break: prefer codewords that cover more already-uncovered words
            # (i.e., higher "density" of coverage). Use random among top.
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = False
            unc_int8 = uncovered.astype(np.int8)
            
        return code

    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        # Initialize to 0
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        # Update cover_count
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Check if removable: all covered words have count > 1
                min_count = np.min(cover_count[mask])
                
                if min_count > 1:
                    # Remove
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try swap
                    # Find unique words
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        # Find candidates that cover all unique words
                        candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                        
                        # Filter out current codewords
                        if len(candidates) > 0:
                            # Exclude current code members
                            current_set = set(current_code)
                            valid_cands = [c for c in candidates if c not in current_set]
                            
                            if len(valid_cands) > 0:
                                new_idx = valid_cands[random.randint(0, len(valid_cands)-1)]
                                
                                # Update cover_count
                                cover_count -= mask
                                cover_count += cover_matrix[new_idx]
                                
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                            
        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    if time.time() < deadline - 15.0:
        for _ in range(100): # Try up to 100 restarts
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
                # Try to improve the new best further
                if time.time() < deadline - 10.0:
                    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
