"""Stronger algorithm for the ternary covering-code task K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np


def _build_cover_matrix(q, n, radius):
    """Build a matrix where M[i][j] = 1 if word i is within distance <= radius of word j."""
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    words_arr = np.array(all_words, dtype=np.int8)
    
    # Compute pairwise Hamming distances efficiently
    # dist[i,j] = number of positions where words[i] != words[j]
    diff = (words_arr[:, :, None] != words_arr[:, None, :]).astype(np.int8)
    dist = diff.sum(axis=1)
    
    cover_matrix = (dist <= radius).astype(np.int8)
    
    return cover_matrix, all_words


def _greedy_cover(cover_matrix, total, time_limit, rng):
    """Greedy set cover using the cover matrix."""
    deadline = time.time() + time_limit
    selected = []
    covered = np.zeros(total, dtype=bool)
    
    while not covered.all():
        if time.time() > deadline - 2:
            break
        
        gains = (cover_matrix[:, ~covered].sum(axis=1))
        best = np.argmax(gains)
        
        if gains[best] == 0:
            break
        
        selected.append(best)
        covered |= cover_matrix[best]
    
    return selected


def _polish_cover(cover_matrix, selected, total, time_limit, rng):
    """Polish the cover by removing redundant codewords."""
    deadline = time.time() + time_limit
    selected = list(selected)
    
    # Pass 1: remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            if time.time() > deadline - 2:
                return selected
            # Check if removing selected[i] still covers everything
            # Compute union of all cover sets except selected[i]
            without_i = np.zeros(total, dtype=bool)
            for j, c in enumerate(selected):
                if j != i:
                    without_i |= cover_matrix[c]
            if without_i.all():
                selected.pop(i)
                changed = True
                break
    
    return selected


def construct(q, n, radius):
    """Construct a covering code of minimum size for the given parameters."""
    start_time = time.time()
    total_time = 120
    safety_margin = 5
    
    total = q ** n
    rng = random.Random(42)
    
    # Build cover matrix
    cover_matrix, all_words = _build_cover_matrix(q, n, radius)
    
    # Greedy construction
    greedy_time = min(30, total_time - safety_margin - 10)
    selected = _greedy_cover(cover_matrix, total, greedy_time, rng)
    
    # Polish
    remaining = total_time - safety_margin - (time.time() - start_time)
    if remaining > 3:
        selected = _polish_cover(cover_matrix, selected, total, remaining, rng)
    
    # Verify
    covered = np.zeros(total, dtype=bool)
    for c in selected:
        covered |= cover_matrix[c]
    
    if not covered.all():
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in all_words]
    
    return [list(all_words[c]) for c in selected]
# EVOLVE-BLOCK-END
