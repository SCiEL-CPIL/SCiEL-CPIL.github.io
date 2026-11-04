"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    """Convert a word (tuple or list) to an integer index in 0..q^n-1."""
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list)."""
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _neighbors_mask(idx, q, n, radius):
    """
    Return a boolean numpy array of length q^n indicating which words
    are within Hamming distance <= radius from the word with index idx.
    For radius=1, this is efficient.
    """
    total = q ** n
    mask = np.zeros(total, dtype=bool)
    
    # The word itself
    mask[idx] = True
    
    if radius >= 1:
        # For each position, change it to each other value
        word = _idx_to_word(idx, q, n)
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                new_idx = _word_to_idx(new_word, q, n)
                mask[new_idx] = True
    
    return mask


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline, start_time):
    """
    Greedy set cover with efficient bitset operations.
    Returns a list of codewords (lists of ints) or None if timeout.
    """
    total = q ** n
    
    # Precompute neighbor masks for all words
    # This is memory intensive but fast for the search
    neighbor_masks = []
    for idx in range(total):
        neighbor_masks.append(_neighbors_mask(idx, q, n, radius))
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # uncovered_count[i] = number of currently uncovered words that codeword i covers
    uncovered_count = np.array([np.sum(~covered & nm) for nm in neighbor_masks], dtype=np.int64)
    
    while time.time() < deadline:
        # Find the candidate with max uncovered_count
        best_idx = int(np.argmax(uncovered_count))
        best_count = int(uncovered_count[best_idx])
        
        if best_count == 0:
            break
        
        # Add best_idx as a codeword
        codewords.append(_idx_to_word(best_idx, q, n))
        
        # Update covered
        covered |= neighbor_masks[best_idx]
        
        # Check if all covered
        if np.all(covered):
            break
        
        # Update uncovered_count: for each candidate, recount uncovered words it covers
        # This is O(total * total) per step which is too slow.
        # Instead, we only update for candidates that are affected.
        # Actually, let's just recompute periodically or use a smarter approach.
        # For now, recompute all (expensive but correct)
        # Optimization: only recompute if we're making progress and time allows
        new_uncovered_count = np.array([np.sum(~covered & nm) for nm in neighbor_masks], dtype=np.int64)
        uncovered_count = new_uncovered_count
    
    if np.all(covered):
        return codewords
    
    return None


def _greedy_prune(codewords, q, n, radius, deadline, start_time):
    """Remove redundant codewords."""
    total = q ** n
    
    # Precompute coverage masks for each codeword
    cover_masks = []
    for cw in codewords:
        idx = _word_to_idx(cw, q, n)
        cover_masks.append(_neighbors_mask(idx, q, n, radius))
    
    # For each word, count how many codewords cover it
    word_cover_count = np.zeros(total, dtype=np.int64)
    for mask in cover_masks:
        word_cover_count += mask.astype(np.int64)
    
    keep = [True] * len(codewords)
    for i in range(len(codewords)):
        if not keep[i]:
            continue
        # Codeword i is removable if all words it covers are covered by at least one other codeword
        # i.e., for all j where cover_masks[i][j] is True, word_cover_count[j] > 1
        covered_by_others = word_cover_count[cover_masks[i]]
        if np.all(covered_by_others > 1):
            keep[i] = False
            # Update word_cover_count
            word_cover_count[cover_masks[i]] -= 1
    
    return [codewords[i] for i in range(len(codewords)) if keep[i]]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try tetracode first as a baseline
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_len = len(tc)
    else:
        best = []
        best_len = float('inf')
    
    # Try greedy cover
    try:
        greedy_result = _greedy_cover(q, n, radius, deadline, start_time)
        if greedy_result is not None:
            # Verify coverage
            total = q ** n
            covered = np.zeros(total, dtype=bool)
            for cw in greedy_result:
                idx = _word_to_idx(cw, q, n)
                covered |= _neighbors_mask(idx, q, n, radius)
            
            if np.all(covered) and len(greedy_result) < best_len:
                best = greedy_result
                best_len = len(best)
                # Try to prune
                if time.time() < deadline - 5.0:
                    pruned = _greedy_prune(best, q, n, radius, deadline, start_time)
                    if pruned:
                        # Verify pruned
                        covered2 = np.zeros(total, dtype=bool)
                        for cw in pruned:
                            idx = _word_to_idx(cw, q, n)
                            covered2 |= _neighbors_mask(idx, q, n, radius)
                        if np.all(covered2) and len(pruned) < best_len:
                            best = pruned
                            best_len = len(best)
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
