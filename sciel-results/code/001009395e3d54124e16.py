import itertools
import time
import random
import numpy as np
from collections import Counter

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute cover sets as numpy boolean arrays for vectorized operations
    # cover[i] is a boolean array of length total_space, True if word j is within radius of word i
    # For q=3, n=7: 2187 * 2187 bits = ~590KB per array, manageable
    # We'll use uint8 arrays for efficient bit counting with numpy
    
    # Precompute word to index and index to word
    # For q=3, n=7, total_space=2187.
    # We can use a simple base-q conversion.
    
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
                    # Create new word
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        # For radius > 1, we would need more logic, but task specifies radius 1
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Precompute reverse cover: for each word, which codewords cover it
    # This is static and helps in the swap phase of local search
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Greedy Set Cover
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            # Optimization: if uncovered is small, we might find it faster? 
            # No, just scan all. 2187 is small.
            for i in range(total_space):
                # Count how many new words this covers
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    # If we cover everything remaining, we can break early?
                    # If gain == uncovered.bit_count(), then we cover all remaining.
                    if gain == uncovered.bit_count():
                        break
                # If best_gain is 1, we can't do better? No, we want max gain.
                # But if best_gain is 1, any other with 1 is equivalent in terms of gain.
                # However, future implications might differ. 
                # For now, standard greedy picks max gain.
                
            if best_idx == -1:
                break # Should not happen if uncovered != 0
            
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Local Search: Remove redundant codewords and perform swaps
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        current_code_set = set(current_code)
        
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
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Phase 1: Removal
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
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            # Phase 2: Swaps (if no removal happened)
            if not improved:
                # Try to swap a codeword with another to enable future removals
                # We look for a codeword c that has some uniquely covered words U.
                # We try to find c_new that covers U.
                # If found, we swap c with c_new.
                
                max_unique_to_cover = 3
                
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    # Find uniquely covered words
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    
                    if unique_mask == 0:
                        # Should have been removed in Phase 1, but if not, remove it
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    if unique_mask.bit_count() > max_unique_to_cover:
                        continue
                    
                    # Search for a replacement
                    # Candidates for c_new must cover all words in unique_mask.
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    
                    # Sort by list size to start with the most constrained
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    
                    # Get candidates from the first word
                    candidates = set(reverse_cover[words_in_unique[0]])
                    
                    # Intersect with others
                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    # Filter out current codeword and any already in code
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        # Score candidates by how many new words they cover
                        # (i.e., words not covered by idx)
                        best_candidate = None
                        best_score = -1
                        cand_list = list(candidates)
                        random.shuffle(cand_list)
                        
                        for cand in cand_list:
                            # Count words covered by cand but not by idx
                            new_cover = (cover_masks[cand] & ~cover_masks[idx]).bit_count()
                            if new_cover > best_score:
                                best_score = new_cover
                                best_candidate = cand
                                if best_score >= 10:  # Early exit if good enough
                                    break
                        
                        new_idx = best_candidate
                        
                        # Perform swap
                        # Remove idx
                        m = cover_masks[idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        
                        # Add new_idx
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        
                        current_code[i] = new_idx
                        current_code_set.remove(idx)
                        current_code_set.add(new_idx)
                        
                        improved = True
                        break
                        
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    
    # Apply local search
    best_code = local_search_reduce(best_code)
    
    # If we have time, we can try a few randomized restarts of the greedy phase
    # to see if we find a better starting point that reduces better.
    # However, greedy is deterministic. We can add random tie-breaking.
    
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
                
            # Pick one randomly from candidates
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    def greedy_cover_weighted():
        """Greedy with random weights to break ties more effectively."""
        uncovered = full_mask
        code = []
        # Assign random weights to each codeword
        weights = [random.random() for _ in range(total_space)]
        
        while uncovered:
            best_score = -1
            best_idx = -1
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > 0:
                    # Use gain as primary, weight as tie-breaker
                    # We want to maximize gain, and among equal gains, prefer higher weight
                    score = gain * 1000.0 + weights[i]
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run a few restarts if time permits
    # The first run already used some time.
    # Let's check time.
    if time.time() < deadline - 20.0:
        # Alternate between different greedy strategies
        for restart in range(15): # Try up to 15 restarts
            if time.time() > deadline - 5.0:
                break
            if restart % 3 == 0:
                cand_code = greedy_cover_randomized()
            elif restart % 3 == 1:
                cand_code = greedy_cover_weighted()
            else:
                # Perturbed greedy: start with greedy, then randomly remove some and re-add
                base_code = greedy_cover()
                # Remove 5-10 random codewords
                n_remove = min(5 + random.randint(0, 5), len(base_code) // 3)
                remove_indices = random.sample(range(len(base_code)), n_remove)
                for ri in sorted(remove_indices, reverse=True):
                    base_code.pop(ri)
                # Re-run local search to see if it can recover
                cand_code = base_code
                # Re-add greedy choices for uncovered words
                # Actually, let's just run local search on the perturbed code
                # But we need to ensure coverage. Let's use a repair step.
                # Simpler: just use the perturbed code as starting point for local search
                # But it might not be a valid cover. Let's stick to valid covers.
                # So let's just use greedy variants.
                cand_code = greedy_cover_randomized()
            
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Final polish: try to remove more codewords with a more aggressive local search
    # Use a different random seed for diversity
    if time.time() < deadline - 3.0:
        random.seed(42)  # Reset seed for reproducibility of this phase
        polished = local_search_reduce(best_code[:])
        if len(polished) < len(best_code):
            best_code = polished
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
