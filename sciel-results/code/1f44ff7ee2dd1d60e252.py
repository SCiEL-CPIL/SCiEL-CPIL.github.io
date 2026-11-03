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
        # Incremental gain tracking: gain[i] = number of still-uncovered words covered by word i
        gain = cm_int8.copy().astype(np.int32)  # initially all uncovered, so gain[i] = popcount(cover_matrix[i])
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            max_gain = np.max(gain)
            if max_gain == 0:
                break
                
            # Find all indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # Random choice among best
            best_idx = int(candidates[random.randint(0, len(candidates)-1)])
            
            code.append(best_idx)
            
            # Update: words newly covered by best_idx become "covered"
            # For each word j that best_idx covers, if j was still uncovered, then
            # for every word i that covers j, gain[i] decreases by 1
            # i.e., gain -= cover_matrix[:, best_idx] restricted to previously-uncovered words
            # More precisely: for each word w in cover_matrix[best_idx] that was uncovered,
            #   gain[cover_matrix[:, w]] -= 1
            # Vectorized: gain -= cover_matrix[:, best_idx] & uncovered  (but we need per-column)
            # Actually: the set of words that best_idx covers AND were uncovered is:
            #   new_covered = cover_matrix[best_idx] & uncovered
            # For each such word w, all codewords i with cover_matrix[i, w] = True lose 1 gain.
            # So: gain -= cover_matrix[:, new_covered_indices].sum(axis=1)
            # But that's expensive. Alternative: 
            #   gain = cm_int8.dot(uncovered)  -- but that's what we're trying to avoid.
            # 
            # Let's use: gain -= (cover_matrix[:, best_idx] & uncovered).sum(axis=1)
            # Wait, that's not right either. Let me think again.
            #
            # gain[i] = sum_w cover_matrix[i,w] * uncovered[w]
            # After setting uncovered[cover_matrix[best_idx]] = 0:
            # new_gain[i] = sum_w cover_matrix[i,w] * new_uncovered[w]
            #             = gain[i] - sum_{w in cover_matrix[best_idx]} cover_matrix[i,w] * uncovered[w]
            #             = gain[i] - sum_{w in cover_matrix[best_idx] & uncovered} cover_matrix[i,w]
            #
            # So: gain -= cover_matrix[:, new_covered].sum(axis=1) where new_covered = cover_matrix[best_idx] & uncovered
            #
            # This is a matrix-vector product restricted to a subset of columns.
            # We can compute: delta = cm_int8[:, new_covered_indices].sum(axis=1)
            # But extracting columns is expensive. Alternative:
            # delta = (cm_int8 * new_covered[None, :]).sum(axis=1)
            # That's a full matrix multiply, same cost as before.
            #
            # For now, let's just do the full dot product. The greedy phase is typically fast enough.
            uncovered[cover_matrix[best_idx]] = 0
            gain = cm_int8.dot(uncovered).astype(np.int32)
            
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
        while improved and time.time() < deadline - 2.0:
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
                            
        # 2-for-1 move: try to remove two codewords and add one that covers their union
        if time.time() < deadline - 3.0:
            improved2 = True
            while improved2 and time.time() < deadline - 3.0:
                improved2 = False
                n_cw = len(current_code)
                if n_cw < 2:
                    break
                random.shuffle(current_code)
                for i in range(n_cw):
                    if time.time() > deadline - 3.0:
                        break
                    for j in range(i+1, n_cw):
                        if time.time() > deadline - 3.0:
                            break
                        idx1 = current_code[i]
                        idx2 = current_code[j]
                        mask1 = cover_matrix[idx1]
                        mask2 = cover_matrix[idx2]
                        
                        # Words that would become uncovered if we remove both
                        # A word w is uniquely covered by idx1 or idx2 if:
                        #   cover_count[w] == 1 and (mask1[w] or mask2[w])
                        # Actually, we need: after removing both, every word still has cover_count >= 1
                        # So: for all w in mask1 | mask2, we need cover_count[w] - (mask1[w] + mask2[w]) >= 1
                        # i.e., cover_count[w] >= 1 + mask1[w] + mask2[w] for all w in mask1 | mask2
                        # And for w not in mask1 | mask2, cover_count[w] >= 1 (already true)
                        
                        union_mask = mask1 | mask2
                        # Check: cover_count[union_mask] >= 1 + (mask1[union_mask] + mask2[union_mask])
                        # mask1 and mask2 are 0/1, so mask1+mask2 is 0,1,or 2
                        # We need: cover_count[w] - mask1[w] - mask2[w] >= 1 for all w in union_mask
                        # i.e., cover_count[w] >= 1 + mask1[w] + mask2[w]
                        needed = 1 + (mask1[union_mask].astype(np.int32) + mask2[union_mask].astype(np.int32))
                        if np.all(cover_count[union_mask] >= needed):
                            # Find a single codeword that covers all of union_mask
                            candidates = np.where(cover_matrix[:, union_mask].all(axis=1))[0]
                            # Exclude current code
                            candidates = candidates[~np.isin(candidates, current_code)]
                            if len(candidates) > 0:
                                new_idx = candidates[random.randint(0, len(candidates)-1)]
                                # Perform the 2-for-1 move
                                cover_count -= mask1
                                cover_count -= mask2
                                cover_count += cover_matrix[new_idx]
                                # Remove both, add new one
                                # Be careful with indices when removing
                                if i < j:
                                    current_code.pop(j)
                                    current_code.pop(i)
                                else:
                                    current_code.pop(i)
                                    current_code.pop(j)
                                current_code.append(int(new_idx))
                                improved2 = True
                                break
                    if improved2:
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
    
    # High-precision polish: keep trying removals until no more can be made
    if time.time() < deadline - 1.0:
        # Rebuild cover_count for the final code
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in best_code:
            cover_count += cover_matrix[idx]
        
        # Try removals repeatedly until stuck
        while time.time() < deadline - 1.0:
            removed = False
            random.shuffle(best_code)
            for i in range(len(best_code)):
                if time.time() > deadline - 1.0:
                    break
                idx = best_code[i]
                mask = cover_matrix[idx]
                if np.min(cover_count[mask]) > 1:
                    cover_count -= mask
                    best_code.pop(i)
                    removed = True
                    break
            if not removed:
                break
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
