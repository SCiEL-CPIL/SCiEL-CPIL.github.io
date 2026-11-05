"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1_bnb()
    else:
        return _generic_greedy_cover(q, n, radius)

def _cover_ternary_6_1_bnb():
    """
    Solve K_3(6,1) using Branch and Bound with Symmetry Breaking and Dominance Pruning.
    """
    alphabet = [0, 1, 2]
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words) # 729
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks (bitsets)
    # cover_mask[i] has bits set for all words within Hamming distance 1 of all_words[i]
    cover_mask = [0] * n_words
    # Precompute conflict masks for packing bound
    # conflict_mask[i] has bits set for all words within Hamming distance 2 of all_words[i]
    conflict_mask = [0] * n_words
    
    # Precomputation
    for i, w in enumerate(all_words):
        # Distance 1
        mask1 = 1 << i
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask1 |= (1 << j)
        cover_mask[i] = mask1
        
        # Distance 2 (for packing bound)
        mask2 = 0
        for j, w2 in enumerate(all_words):
            d = sum(1 for a, b in zip(w, w2) if a != b)
            if d <= 2:
                mask2 |= (1 << j)
        conflict_mask[i] = mask2

    full_mask = (1 << n_words) - 1
    
    # Symmetry Breaking: Fix first codeword to index 0 (000000)
    first_cw = 0
    initial_uncovered = full_mask & ~cover_mask[first_cw]
    
    def greedy_packing_bound(uncovered_mask):
        """
        Estimate lower bound on number of codewords needed to cover 'uncovered_mask'.
        Uses greedy packing: find max set of words in 'uncovered_mask' with pairwise dist >= 3.
        """
        remaining = uncovered_mask
        count = 0
        while remaining:
            # Pick a word from remaining (lowest set bit)
            u_idx = (remaining & -remaining).bit_length() - 1
            count += 1
            # Remove u and all words within dist 2
            remaining &= ~conflict_mask[u_idx]
        return count

    def greedy_solve(uncovered_mask, current_cws):
        """
        Greedy set cover to get an initial upper bound.
        """
        cws = list(current_cws)
        unc = uncovered_mask
        while unc:
            best_cw = -1
            best_count = -1
            for i in range(n_words):
                if i in cws: continue
                count = (cover_mask[i] & unc).bit_count()
                if count > best_count:
                    best_count = count
                    best_cw = i
            if best_cw == -1:
                break
            cws.append(best_cw)
            unc &= ~cover_mask[best_cw]
        return cws

    # Get initial solution
    initial_cws = [0]
    best_cws = greedy_solve(initial_uncovered, initial_cws)
    best_size = len(best_cws)
    
    start_time = time.time()
    timeout = 110
    
    def search(uncovered_mask, current_cws, current_size):
        nonlocal best_cws, best_size
        
        if time.time() - start_time > timeout:
            return
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_cws = list(current_cws)
            return
        
        # Lower bound
        lb = greedy_packing_bound(uncovered_mask)
        if current_size + lb >= best_size:
            return
            
        # Branching
        # Pick an uncovered word 'e' to force coverage.
        e_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Candidates: all i such that e_idx is in cover_mask[i] and i not in current_cws
        candidates = []
        for i in range(n_words):
            if i in current_cws:
                continue
            if cover_mask[i] & (1 << e_idx):
                new_cov_mask = cover_mask[i] & uncovered_mask
                candidates.append((new_cov_mask, i))
                
        if not candidates:
            return

        # Dominance Pruning
        # If Cover(A) subset Cover(B), A is dominated by B.
        # candidates list contains (mask, idx)
        non_dominated = []
        for k, (mask_a, idx_a) in enumerate(candidates):
            dominated = False
            for j, (mask_b, idx_b) in enumerate(candidates):
                if k == j: continue
                # Check if mask_a is subset of mask_b
                if (mask_a & ~mask_b) == 0:
                    # If strictly subset or equal, A is dominated
                    # If equal, we can keep one. Let's keep the one with smaller index for determinism?
                    # Or just mark as dominated if j < k to break ties?
                    # If mask_a == mask_b, they are equivalent. We can skip one.
                    if mask_a == mask_b:
                        if idx_a > idx_b:
                            dominated = True
                            break
                    else:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append((mask_a, idx_a))
                
        # Sort by coverage size descending
        non_dominated.sort(key=lambda x: x[0].bit_count(), reverse=True)
        
        for mask, i in non_dominated:
            new_uncovered = uncovered_mask & ~mask
            search(new_uncovered, current_cws + [i], current_size + 1)

    search(initial_uncovered, [0], 1)
    
    # Convert indices to words
    result = [list(all_words[i]) for i in best_cws]
    return result

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    covered_sets = []
    for w in all_words:
        covered = []
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.append(candidate)
        covered_sets.append(set(word_to_idx[c] for c in covered))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            if best_count == len(uncovered):
                break
        
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
    
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
