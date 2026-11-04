# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def _ball_indices(word, q, n, radius):
    """Return a frozenset of indices (integers 0..q^n-1) within Hamming distance radius of word."""
    word = np.array(word, dtype=np.int8)
    indices = [int(np.ravel_multi_index(tuple(word), (q,) * n))]
    if radius >= 1:
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                w2 = word.copy()
                w2[pos] = val
                indices.append(int(np.ravel_multi_index(tuple(w2), (q,) * n)))
    return frozenset(indices)


def _tetracode(q, n):
    """Product construction: ternary tetracode (9 words of length 4) x all words on remaining coords."""
    if n < 4:
        # Fallback: all words
        return [list(w) for w in __import__('itertools').product(range(q), repeat=n)]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover to find a small covering code."""
    total = q ** n
    # Precompute all balls
    all_balls = []
    for idx in range(total):
        word = np.unravel_index(idx, (q,) * n)
        all_balls.append(_ball_indices(list(word), q, n, radius))
    
    uncovered = set(range(total))
    codewords = []
    
    while uncovered and time.time() < deadline:
        # Find the word whose ball covers the most uncovered points
        best_idx = -1
        best_count = -1
        for idx in range(total):
            count = len(all_balls[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
            if count == 0:
                continue
        if best_count <= 0:
            break
        word = list(np.unravel_index(best_idx, (q,) * n))
        codewords.append(word)
        uncovered -= all_balls[best_idx]
    
    return codewords


def _greedy_cover_fast(q, n, radius, deadline):
    """Faster greedy using numpy for coverage counting."""
    total = q ** n
    # Build a matrix: rows = words, cols = points, 1 if point is in ball of word
    # This is too large for n=6 (729x729=531k, actually fine)
    
    # Precompute ball membership as bitsets
    balls = np.zeros((total, total), dtype=bool)
    for idx in range(total):
        word = list(np.unravel_index(idx, (q,) * n))
        for p in _ball_indices(word, q, n, radius):
            balls[idx, p] = True
    
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while uncovered.any() and time.time() < deadline:
        # Count uncovered points covered by each candidate
        counts = balls[:, uncovered].sum(axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        word = list(np.unravel_index(best_idx, (q,) * n))
        codewords.append(word)
        uncovered &= ~balls[best_idx]
    
    return codewords


def _local_search_remove(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords."""
    total = q ** n
    # Precompute balls for current codewords
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Check if removing codewords[i] still covers everything
            remaining = [j for j in range(len(codewords)) if j != i]
            covered = set()
            for j in remaining:
                covered |= ball_sets[j]
            if len(covered) == total:
                # Remove codeword i
                codewords.pop(i)
                ball_sets.pop(i)
                improved = True
                break
    
    return codewords


def _swap_local_search(codewords, q, n, radius, deadline):
    """Try swapping one codeword for another to improve coverage."""
    total = q ** n
    ball_sets = [_ball_indices(cw, q, n, radius) for cw in codewords]
    
    # Compute current coverage
    all_covered = set()
    for bs in ball_sets:
        all_covered |= bs
    
    if len(all_covered) < total:
        # Not a valid cover, try to fix
        pass
    
    # Try single swaps
    for i in range(len(codewords)):
        if time.time() > deadline:
            break
        # Try replacing codeword i with each possible word
        current_ball = ball_sets[i]
        # The set of points only covered by codeword i
        others_covered = set()
        for j in range(len(codewords)):
            if j != i:
                others_covered |= ball_sets[j]
        
        # Points that need to be covered by the new word
        must_cover = all_covered - others_covered
        
        # Try candidate words
        for idx in range(total):
            candidate_ball = _ball_indices(list(np.unravel_index(idx, (q,) * n)), q, n, radius)
            if must_cover <= candidate_ball:
                # This swap maintains coverage
                new_ball = candidate_ball
                # Check if this reduces total covered area (overlap)
                old_total = len(all_covered)
                new_covered = others_covered | candidate_ball
                if len(new_covered) == total:
                    # Valid swap; prefer one that minimizes overlap
                    overlap = len(candidate_ball & others_covered)
                    if overlap < len(current_ball & others_covered):
                        codewords[i] = list(np.unravel_index(idx, (q,) * n))
                        ball_sets[i] = candidate_ball
                        all_covered = others_covered | candidate_ball
                        break
    
    return codewords


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # 115s safety margin out of 150s
    
    # Start with tetracode product construction
    initial = _tetracode(q, n)
    best = [tuple(cw) for cw in initial]
    
    # Try greedy cover
    if time.time() < deadline - 40:
        try:
            greedy = _greedy_cover_fast(q, n, radius, deadline - 40)
            if len(greedy) < len(best):
                best = [tuple(cw) for cw in greedy]
        except Exception:
            pass
    
    # Local search to remove redundant codewords
    if time.time() < deadline - 5:
        best = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 5)
        best = [tuple(cw) for cw in best]
    
    # Try swap local search for further improvement
    if time.time() < deadline - 2:
        best = _swap_local_search([list(cw) for cw in best], q, n, radius, deadline - 2)
        best = [tuple(cw) for cw in best]
    
    # Final removal pass
    if time.time() < deadline - 1:
        best = _local_search_remove([list(cw) for cw in best], q, n, radius, deadline - 1)
        best = [tuple(cw) for cw in best]
    
    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
