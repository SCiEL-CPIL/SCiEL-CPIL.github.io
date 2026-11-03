import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before the 150s kill
    
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
    
    # Build cover matrix
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    cover_matrix[i, j] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Precompute neighbor lists for each word (words within radius)
    # This helps in fast gain calculation and local search
    # neighbors[i] is a list of indices j such that cover_matrix[i, j] is True
    # Actually, for the swap operation, we need to know which words are uniquely covered.
    # Let's stick to the matrix approach for now but optimize the inner loops.
    
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gain for each codeword
            gains = cm_int8.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best to allow diversity in restarts
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(int(best_idx))
            uncovered[cover_matrix[best_idx]] = 0
            
        return code

    def local_search_reduce(code, max_time=None):
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
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
                if time.time() > deadline - 2.0:
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
                    unique_mask = mask & (cover_count == 1)
                    
                    if np.any(unique_mask):
                        unique_indices = np.where(unique_mask)[0]
                        
                        if len(unique_indices) > 0:
                            # Create exclusion mask
                            exclude_mask = np.zeros(total_space, dtype=bool)
                            for j in current_code:
                                exclude_mask[j] = True
                        
                            # Find candidates that cover all unique words
                            # We need j such that cover_matrix[j, unique_indices] are all True
                            # This is equivalent to: sum(cover_matrix[j, unique_indices]) == len(unique_indices)
                            # Or: cover_matrix[:, unique_indices].all(axis=1)
                            
                            # Optimization: Instead of checking all candidates, we can be smarter.
                            # But for now, let's use the vectorized check.
                            # cover_matrix[:, unique_indices] has shape (N, K)
                            # .all(axis=1) gives a boolean array of length N
                            
                            # To speed up, we can filter candidates first?
                            # Maybe just check candidates that are not in current_code.
                            
                            candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                            candidates = candidates[~exclude_mask[candidates]]
                        
                            if len(candidates) > 0:
                                new_idx = candidates[random.randint(0, len(candidates)-1)]
                                
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
    # We use a while loop with a time check to maximize the number of restarts
    while time.time() < deadline - 10.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search on best_code with remaining time
    # This might find further reductions if the previous local search stopped early due to time
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
