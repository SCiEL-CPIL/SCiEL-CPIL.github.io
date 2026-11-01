# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using bitmask-based local search.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    total_words = 3 ** n
    all_indices = range(total_words)
    
    # Precompute neighbors mask for each word
    # Word i corresponds to tuple of base-3 digits
    neighbors_mask = [0] * total_words
    for i in range(total_words):
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % 3
            temp //= 3
        
        mask = 0
        # The word itself
        mask |= (1 << i)
        
        # Neighbors: change one digit
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = w[:]
                new_w[pos] = new_val
                # Convert back to index
                idx = 0
                base = 1
                for k in range(n):
                    idx += new_w[k] * base
                    base *= 3
                mask |= (1 << idx)
        neighbors_mask[i] = mask

    full_mask = (1 << total_words) - 1

    def popcount(x):
        return bin(x).count("1")

    def greedy_cover():
        uncovered = full_mask
        code = []
        # To allow randomization, we can shuffle the list of candidates or add noise
        # Here we just pick the best, breaking ties randomly
        while uncovered:
            best_c = -1
            best_count = -1
            best_candidates = []
            for c in all_indices:
                if c in code:
                    continue
                count = popcount(neighbors_mask[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_candidates = [c]
                elif count == best_count:
                    best_candidates.append(c)
            
            if best_count <= 0:
                break # Should not happen if space is finite and we pick valid words
                
            # Pick randomly from best candidates to diversify
            best_c = random.choice(best_candidates)
            code.append(best_c)
            uncovered &= ~neighbors_mask[best_c]
            
        return code

    def local_search(initial_code):
        code_set = set(initial_code)
        current_mask = 0
        for c in code_set:
            current_mask |= neighbors_mask[c]
            
        if current_mask != full_mask:
            return None, None

        improved = True
        while improved:
            improved = False
            
            # Phase 1: Removal
            for c in list(code_set):
                new_mask = current_mask & ~neighbors_mask[c]
                if new_mask == full_mask:
                    code_set.remove(c)
                    current_mask = new_mask
                    improved = True
                    break # Restart to check if more can be removed
            
            # Phase 2: Swap
            if not improved:
                for c_old in list(code_set):
                    mask_wo_old = current_mask & ~neighbors_mask[c_old]
                    # Words that become uncovered if c_old is removed
                    needed = full_mask & ~mask_wo_old
                    if needed == 0:
                        # This should have been caught in removal, but just in case
                        code_set.remove(c_old)
                        current_mask = mask_wo_old
                        improved = True
                        break
                    
                    # Find a c_new that covers 'needed'
                    # Optimization: We can iterate through all indices.
                    # To speed up, we could precompute which words cover which 'needed' patterns? No, needed changes.
                    # Just iterate.
                    found_swap = False
                    for c_new in all_indices:
                        if c_new in code_set:
                            continue
                        # Check if c_new covers all 'needed' words
                        if (neighbors_mask[c_new] & needed) == needed:
                            # Perform swap
                            code_set.remove(c_old)
                            code_set.add(c_new)
                            current_mask = mask_wo_old | neighbors_mask[c_new]
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                        
        return len(code_set), list(code_set)

    best_size = float('inf')
    best_code = None
    
    # Run multiple restarts
    # Time limit is 120s. Local search is fast.
    # Greedy is O(N^2) roughly. 729^2 = 531k ops. Very fast.
    # Local search is O(K * N) per pass.
    
    start_time = time.time()
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() - start_time > 110:
            break
            
        init_code = greedy_cover()
        size, code_list = local_search(init_code)
        
        if size is not None and size < best_size:
            best_size = size
            best_code = code_list
            
        # Early stop if we hit the theoretical lower bound or known best
        if best_size <= 73:
            break

    if best_code is None:
        # Fallback to a single greedy result if something went wrong
        return [list(itertools.product(range(3), repeat=6))[c] for c in greedy_cover()]
        
    # Convert indices back to lists
    result = []
    for idx in best_code:
        w = [0] * n
        temp = idx
        for j in range(n):
            w[j] = temp % 3
            temp //= 3
        result.append(w)
        
    return result

def _generic_greedy(q, n, radius):
    # Fallback for non-specific cases
    all_words = list(itertools.product(range(q), repeat=n))
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if tuple(w) in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
