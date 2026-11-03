import itertools
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
    
    # Precompute cover masks
    # mask[i] is a bitmask of indices of words within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        # For radius > 1, we would need more logic, but task specifies radius 1
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Greedy Set Cover
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        return code

    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Local Search: Remove redundant codewords and perform swaps
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        
        # Build cover_count array: cover_count[w] = number of codewords covering word w
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    # Decrement cover_count and remove
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a swap
                    # Find words uniquely covered by idx
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= low
                        m ^= low
                    
                    if unique_mask:
                        # Find a candidate j not in current_code that covers all words in unique_mask
                        # Optimization: Only check candidates that cover at least one word in unique_mask?
                        # For now, iterate all, but break early if found.
                        # We can also randomize the order of candidates to escape local minima.
                        candidates = []
                        for j in range(total_space):
                            if j in current_code:
                                continue
                            if (cover_masks[j] & unique_mask) == unique_mask:
                                candidates.append(j)
                        
                        if candidates:
                            # Choose a random candidate
                            new_idx = random.choice(candidates)
                            
                            # Update cover_count
                            # Remove old idx
                            m = mask
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                            
                            # Add new idx
                            m_new = cover_masks[new_idx]
                            while m_new:
                                low = m_new & (-m_new)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m_new ^= low
                            
                            current_code[i] = new_idx
                            improved = True
                            break
            
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    best_code = local_search_reduce(best_code)
    
    # Run a few restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(20): # Try up to 20 restarts
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
