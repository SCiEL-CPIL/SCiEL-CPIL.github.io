import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
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
    # cover_matrix[i, j] is True if word j is within Hamming distance <= radius of word i
    cover_matrix = np.zeros((total_space, total_space), dtype=bool)
    np.fill_diagonal(cover_matrix, True)
    
    if radius >= 1:
        # For radius 1, we can efficiently compute neighbors
        # Precompute powers for index calculation
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        
        for i in range(total_space):
            w = all_words[i]
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    j = i + (val - orig_val) * powers[pos]
                    cover_matrix[i, j] = True
    elif radius > 1:
        # Fallback for radius > 1: compute Hamming distance for all pairs
        for i in range(total_space):
            wi = all_words[i]
            for j in range(i, total_space):
                wj = all_words[j]
                dist = sum(1 for k in range(n) if wi[k] != wj[k])
                if dist <= radius:
                    cover_matrix[i, j] = True
                    cover_matrix[j, i] = True

    # Precompute int8 version of cover matrix for fast dot products
    cm_int8 = cover_matrix.astype(np.int8)
    
    # Precompute neighbor lists
    neighbors = [np.where(cover_matrix[i])[0] for i in range(total_space)]
    
    def greedy_cover_randomized():
        uncovered = np.ones(total_space, dtype=np.int8)
        code = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gains: how many uncovered words each codeword covers
            gains = cm_int8.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random choice among best
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
                            candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                            candidates = candidates[~exclude_mask[candidates]]
                        
                            if len(candidates) > 0:
                                # Among candidates, pick one that covers the most other words
                                # or just pick randomly for diversity
                                new_idx = candidates[random.randint(0, len(candidates)-1)]
                                
                                # Update cover_count
                                cover_count -= mask
                                cover_count += cover_matrix[new_idx]
                                
                                current_code[i] = int(new_idx)
                                improved = True
                                break
                            
        return current_code

    def local_search_swap(code, max_time=None):
        """
        Additional local search: try to replace a codeword with another that
        maintains coverage while potentially allowing further removals.
        """
        if not code:
            return code
        
        if max_time is None:
            max_time = deadline - time.time()
        
        current_code = code[:]
        
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in current_code:
            cover_count += cover_matrix[idx]
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline - 2.0:
                    return current_code
                
                idx = current_code[i]
                mask = cover_matrix[idx]
                
                # Find words uniquely covered by this codeword
                unique_mask = mask & (cover_count == 1)
                
                if np.any(unique_mask):
                    unique_indices = np.where(unique_mask)[0]
                    
                    # Find all codewords that cover all unique words
                    candidates = np.where(cover_matrix[:, unique_indices].all(axis=1))[0]
                    
                    # Exclude current codeword
                    candidates = candidates[candidates != idx]
                    
                    if len(candidates) > 0:
                        # For each candidate, compute how many additional words it covers
                        # that are currently covered by other codewords (redundant coverage)
                        best_candidate = -1
                        best_score = -1
                        
                        for cand in candidates:
                            cand_mask = cover_matrix[cand]
                            # Score: number of words covered by cand that have cover_count > 1
                            # This means we're not losing unique coverage, and gaining redundancy elsewhere
                            # Actually, we want to minimize the "loss" of unique coverage
                            # Since cand covers all unique words, no unique coverage is lost.
                            # The benefit is if cand covers words that are currently over-covered.
                            # Let's just pick randomly for now, or pick the one with max overlap with current code
                            overlap = np.sum(cand_mask & (cover_count > 1))
                            if overlap > best_score:
                                best_score = overlap
                                best_candidate = cand
                        
                        if best_candidate >= 0:
                            # Perform swap
                            cover_count -= mask
                            cover_count += cover_matrix[best_candidate]
                            current_code[i] = int(best_candidate)
                            improved = True
                            break
                else:
                    # No unique words, so this codeword can be removed
                    cover_count -= mask
                    current_code.pop(i)
                    improved = True
                    break
        
        return current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    best_code = local_search_swap(best_code)
    
    # Run restarts if time permits
    while time.time() < deadline - 15.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        cand_code = local_search_swap(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search on best_code with remaining time
    best_code = local_search_reduce(best_code)
    best_code = local_search_swap(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
