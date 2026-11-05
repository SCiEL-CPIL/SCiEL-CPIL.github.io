"""Construct a small ternary covering code of length 8, radius 1.

Strategy:
- Use integer bitmask representation for fast coverage operations.
- Greedy set cover with randomized restarts.
- Aggressive local search: removal, swap, and 2-exchange moves.
"""

import itertools
import random
import time
import numpy as np

# Precomputation
def _precompute():
    """Precompute all words, their indices, and coverage masks as integers."""
    n = 8
    q = 3
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total = len(all_words)  # 6561
    
    # For each word, compute coverage mask as a Python int (bitset)
    cover_masks = []
    for w in all_words:
        mask = 0
        mask |= (1 << word_to_idx[w])
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                w2 = list(w)
                w2[pos] = alt
                w2 = tuple(w2)
                mask |= (1 << word_to_idx[tuple(w2)])
        cover_masks.append(mask)
    
    return all_words, word_to_idx, cover_masks

_all_words = None
_word_to_idx = None
_cover_masks = None
_total_words = None
_full_mask = None

def _ensure_precomputed():
    global _all_words, _word_to_idx, _cover_masks, _total_words, _full_mask
    if _all_words is None:
        _all_words, _word_to_idx, _cover_masks = _precompute()
        _total_words = len(_all_words)
        _full_mask = (1 << _total_words) - 1

def construct(q, n, radius):
    """Return a list of codewords (each a list of n ints in 0..q-1) covering
    all of {0..q-1}^n within Hamming distance radius."""
    _ensure_precomputed()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: trivial construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = _all_words
    cover_masks = _cover_masks
    total_words = _total_words
    full_mask = _full_mask
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_solution = None
    best_count = float('inf')
    
    # Multiple randomized greedy restarts with local search
    for restart in range(50):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Greedy set cover with slight randomization
        uncovered = full_mask
        selected = set()
        
        # Precompute list of indices for faster iteration
        indices = list(range(total_words))
        
        while uncovered:
            best_idx = -1
            best_count = 0
            # Randomize tie-breaking by shuffling if many ties
            # For speed, just do a linear scan with random tie-break
            for i in indices:
                if i in selected:
                    continue
                overlap = bin(cover_masks[i] & uncovered).count('1')
                if overlap > best_count or (overlap == best_count and best_idx >= 0 and random.random() < 0.1):
                    best_count = overlap
                    best_idx = i
                if best_count == 0:
                    break
            if best_idx < 0:
                break
            selected.add(best_idx)
            uncovered &= ~cover_masks[best_idx]
        
        # Local search: iteratively try to remove codewords
        changed = True
        while changed:
            changed = False
            for idx in list(selected):
                if time.time() - start_time > time_limit - 20:
                    break
                # Check if removing idx still covers everything
                remaining_cover = 0
                for j in selected:
                    if j != idx:
                        remaining_cover |= cover_masks[j]
                if (remaining_cover | cover_masks[idx]) == full_mask and remaining_cover == full_mask:
                    selected.discard(idx)
                    changed = True
        
        # Local search: try swaps - replace one codeword with another to enable removals
        improved = True
        while improved:
            improved = False
            for idx in list(selected):
                if time.time() - start_time > time_limit - 15:
                    break
                # Try removing idx and finding a replacement that allows removing others
                selected.discard(idx)
                # Find uncovered
                cur_cover = 0
                for j in selected:
                    cur_cover |= cover_masks[j]
                uncovered = full_mask & ~cur_cover
                if uncovered == 0:
                    improved = True
                    continue
                # Find best replacement
                best_repl = -1
                best_gain = 0
                for i in range(total_words):
                    if i in selected:
                        continue
                    gain = bin(cover_masks[i] & uncovered).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_repl = i
                if best_repl >= 0 and best_gain > 0:
                    selected.add(best_repl)
                    # Now try to remove more
                    changed2 = True
                    while changed2:
                        changed2 = False
                        for j in list(selected):
                            if time.time() - start_time > time_limit - 10:
                                break
                            sel_without_j = selected - {j}
                            cover_without_j = 0
                            for k in sel_without_j:
                                cover_without_j |= cover_masks[k]
                            if cover_without_j == full_mask:
                                selected.discard(j)
                                changed2 = True
                    improved = True
                    break
        
        count = len(selected)
        if count < best_count:
            best_count = count
            best_solution = set(selected)
        
        # If we've found a very good solution, break early
        if best_count <= 410:
            break
    
    if best_solution is None:
        # Fallback
        best_solution = set(range(0, total_words, 17))
    
    result = [list(all_words[idx]) for idx in sorted(best_solution)]
    return result
