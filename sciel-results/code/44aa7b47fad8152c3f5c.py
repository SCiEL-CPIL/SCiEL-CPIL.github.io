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
    
    # Build cover matrix using vectorized NumPy operations
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    
    # Diagonal is True
    np.fill_diagonal(cover_matrix, True)
    
    # For radius 1: for each position and value change, update matrix in bulk
    if radius >= 1:
        # Precompute powers
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        
        # For each position
        for pos in range(n):
            power = powers[pos]
            
            # For each pair of different values
            for val1 in range(q):
                for val2 in range(q):
                    if val1 == val2:
                        continue
                    
                    # Find all indices where position pos has value val1
                    # idx = sum(w[k] * q^(n-1-k))
                    # w[pos] = val1
                    # So idx mod (q^(n-pos)) has digit val1 at position (n-1-pos)
                    # Actually, let's think in terms of the index structure
                    # idx = ... + val1 * q^(n-1-pos) + ...
                    
                    # We can construct these indices by iterating over all combinations of other positions
                    # But that's complex. Let's use a different approach.
                    
                    # Alternative: for each index i, check if w[i][pos] == val1
                    # Then j = i + (val2 - val1) * power
                    # And set cover_matrix[i, j] = True
                    
                    # This is still a loop, but we can vectorize it.
                    # Let's create a boolean mask for indices with w[pos] == val1
                    # idx mod (q * power) // power == val1
                    # Actually: idx // power % q == val1
                    
                    mask = (np.arange(total_space) // power) % q == val1
                    valid_indices = np.where(mask)[0]
                    
                    if len(valid_indices) > 0:
                        # For each valid index i, j = i + (val2 - val1) * power
                        # Set cover_matrix[i, j] = True
                        targets = valid_indices + (val2 - val1) * power
                        # Ensure targets are in range (they should be since we're just changing one digit)
                        valid_targets = targets[(targets >= 0) & (targets < total_space)]
                        valid_src = valid_indices[(targets >= 0) & (targets < total_space)]
                        
                        if len(valid_src) > 0:
                            cover_matrix[valid_src, valid_targets] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Also precompute for each codeword, the list of words it covers (as a boolean array)
    # This is already cover_matrix
    
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
            
            # Tie-break: use random among top
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
                        # A candidate c covers all unique words if cover_matrix[c, u] is True for all u in unique_indices
                        # This is equivalent to: np.all(cover_matrix[c, unique_indices])
                        
                        # Vectorized check: for each candidate c, check if it covers all unique_indices
                        # cover_matrix[:, unique_indices] has shape (total_space, len(unique_indices))
                        # We want rows where all entries are True
                        if len(unique_indices) > 0:
                            candidates_mask = np.all(cover_matrix[:, unique_indices], axis=1)
                            candidates = np.where(candidates_mask)[0]
                        else:
                            candidates = np.arange(total_space)
                        
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
