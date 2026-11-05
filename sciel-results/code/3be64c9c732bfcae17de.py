import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Handle the specific case q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)
    
    # Generic fallback: return all words (valid but large)
    all_words = list(itertools.product(range(q), repeat=n))
    return [list(w) for w in all_words]


def _solve_ternary_6_1(deadline):
    q, n = 3, 6
    total_words = q ** n  # 729
    
    # Precompute: for each word, its index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors for all words (word + all at distance 1)
    # neighbors[i] = set of word indices covered by codeword i
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        nbrs = {i}
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w2 = list(w)
                w2[pos] = val
                nbrs.add(word_to_idx(tuple(w2)))
        neighbors.append(nbrs)
    
    # Build incidence: cover_matrix[c][w] = True if codeword c covers word w
    # Use numpy bitset approach: for each codeword, a 729-bit mask
    # 729 bits = 12 64-bit words or use Python int bitmasks
    
    # For each possible codeword (729 options), precompute its coverage bitmask
    # We'll use Python integers as bitmasks (729 bits fits easily)
    cov_masks = [0] * total_words
    for c in range(total_words):
        mask = 0
        for w in neighbors[c]:
            mask |= (1 << w)
        cov_masks[c] = mask
    
    # Start with the direct-sum construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=2):
            initial_code.append(tuple(h) + tail)
    
    initial_indices = [word_to_idx(c) for c in initial_code]
    
    # Quick greedy prune on initial code
    best_code = list(initial_code)
    best_indices = list(initial_indices)
    
    # Greedy removal
    active = set(range(len(initial_code)))
    for _ in range(10):
        removed_any = False
        for ci in list(active):
            if time.time() > deadline - 10:
                break
            # Check if removing ci leaves all covered words still covered
            # covered_by[ci] words must each be covered by at least one other active codeword
            is_redundant = True
            for w in neighbors[initial_indices[ci]]:
                found = False
                for cj in active:
                    if cj == ci:
                        continue
                    if (cov_masks[initial_indices[cj]] >> w) & 1:
                        found = True
                        break
                if not found:
                    is_redundant = False
                    break
            if is_redundant:
                active.remove(ci)
                removed_any = True
        if not removed_any:
            break
    
    pruned = [initial_code[i] for i in active]
    if len(pruned) < len(best_code):
        best_code = pruned
        best_indices = [initial_indices[i] for i in active]
    
    # Now do local search: try swapping a codeword with another to reduce size
    # Strategy: iteratively try to remove one codeword and add a different one
    # that covers more previously-uncovered ground, netting a size reduction
    
    # Better approach: randomized greedy set cover with many restarts
    # Use bitmask operations for speed
    
    full_mask = (1 << total_words) - 1
    
    def verify(indices):
        """Check if indices cover all words"""
        mask = 0
        for idx in indices:
            mask |= cov_masks[idx]
        return mask == full_mask
    
    def greedy_set_cover(start_indices, max_iter=50):
        """Start from start_indices, greedily remove redundant, then add useful ones"""
        current = set(start_indices)
        
        # Phase 1: Remove redundant
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            removed = False
            for ci in list(current):
                is_redundant = True
                for w in neighbors[ci]:
                    found = False
                    for cj in current:
                        if cj == ci:
                            continue
                        if (cov_masks[cj] >> w) & 1:
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                if is_redundant:
                    current.discard(ci)
                    removed = True
            if not removed:
                break
        
        # Phase 2: Try to add codewords that enable removal of multiple others
        # This is complex; instead, try swap: remove one, see if we can add one that covers the gap
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            # Compute what's uniquely covered by each codeword
            # For each word, count how many codewords cover it
            # If a word is covered by exactly 1 codeword, that codeword can't be removed
            # Try to find a codeword that can be removed, then find a replacement
            best_swap = None
            for ci in list(current):
                # What words does ci uniquely cover?
                unique_words = []
                for w in neighbors[ci]:
                    count = 0
                    for cj in current:
                        if (cov_masks[cj] >> w) & 1:
                            count += 1
                            if count > 1:
                                break
                    if count == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    continue  # ci is redundant, just remove it
                
                # Can we find a codeword not in current that covers all unique_words?
                for cj in range(total_words):
                    if cj in current:
                        continue
                    covers_all = True
                    for w in unique_words:
                        if not ((cov_masks[cj] >> w) & 1):
                            covers_all = False
                            break
                    if covers_all:
                        # Check: after removing ci and adding cj, is everything still covered?
                        new_set = (current - {ci}) | {cj}
                        if verify(list(new_set)):
                            best_swap = (ci, cj)
                            break
                
                if best_swap:
                    break
            
            if best_swap:
                ci, cj = best_swap
                current.discard(ci)
                current.add(cj)
        
        return list(current)
    
    # Run multiple random restarts with different initial orderings
    best_size = len(best_code)
    best_final = best_code
    
    # Try randomized greedy with different random seeds
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        
        # Create a random permutation of the initial code
        shuffled = initial_code[:]
        random.shuffle(shuffled)
        shuffled_idx = [word_to_idx(c) for c in shuffled]
        
        result = greedy_set_cover(shuffled_idx, max_iter=30)
        
        if len(result) < best_size:
            best_size = len(result)
            best_final = [idx_to_word(i) for i in result]
            if best_size <= 73:
                break
    
    # Additional: try starting from scratch with randomized greedy set cover
    for trial in range(10):
        if time.time() > deadline - 15:
            break
        
        # Randomized greedy set cover from empty
        uncovered = full_mask
        code = []
        remaining = list(range(total_words))
        random.shuffle(remaining)
        
        while uncovered and remaining and time.time() < deadline - 15:
            # Find the codeword that covers the most uncovered words
            best_c = -1
            best_count = 0
            for c in remaining:
                count = bin(cov_masks[c] & uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_c = c
                if time.time() > deadline - 15:
                    break
            
            if best_c == -1 or best_count == 0:
                break
            
            code.append(best_c)
            uncovered &= ~cov_masks[best_c]
            remaining.remove(best_c)
        
        if uncovered == 0 and len(code) < best_size:
            best_size = len(code)
            best_final = [idx_to_word(i) for i in code]
            if best_size <= 73:
                break
    
    # Final verification
    if verify([word_to_idx(c) for c in best_final]):
        return [list(c) for c in best_final]
    
    # Fallback: return the original 81-word construction
    return [list(c) for c in initial_code]
