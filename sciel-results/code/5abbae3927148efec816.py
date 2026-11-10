"""Improved covering-code search for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 150.0  # wall-clock seconds
_MARGIN = 2.0      # safety margin


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _within_radius(word, code, radius):
    return any(_hamming(word, c) <= radius for c in code)


def _is_covering(code, q, n, radius):
    # Full check over all q^n words
    for word in itertools.product(range(q), repeat=n):
        if not _within_radius(word, code, radius):
            return False
    return True


def _precompute_neighborhoods(q, n, radius):
    """Precompute for each word, the list of words within radius."""
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighborhoods = []
    for w in all_words:
        neighbors = []
        for v in all_words:
            if sum(a != b for a, b in zip(w, v)) <= radius:
                neighbors.append(word_to_idx[v])
        neighborhoods.append(neighbors)
    return all_words, word_to_idx, neighborhoods


def _base_tetracode_sum(q, n, radius):
    # Direct sum of ternary tetracode with full space on remaining coords
    if q != 3 or n < 4:
        # Fallback: greedy
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    words = [h + list(tail) for h in tetracode
             for tail in itertools.product(range(3), repeat=n - 4)]
    return words


def _greedy_cover_fast(all_words, word_to_idx, neighborhoods, q, n, radius, start_code=None, time_limit=30.0):
    """Fast greedy set-cover using precomputed neighborhoods."""
    t0 = time.time()
    N = len(all_words)
    if start_code is None:
        start_code = []

    covered = np.zeros(N, dtype=bool)
    for c in start_code:
        idx = word_to_idx[tuple(c)]
        covered[neighborhoods[idx]] = True

    code = [list(c) for c in start_code]

    while not covered.all() and (time.time() - t0) < time_limit:
        best_gain = 0
        best_idx = -1
        uncovered_count = N - covered.sum()
        
        # Sample a subset of candidates for speed
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for idx in candidates:
            if covered.all():
                break
            gain = sum(1 for j in neighborhoods[idx] if not covered[j])
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
            if (time.time() - t0) > time_limit:
                break

        if best_gain == 0 or best_idx < 0:
            break

        code.append(list(all_words[best_idx]))
        covered[neighborhoods[best_idx]] = True

    return code


def _local_search_improve_fast(code, all_words, word_to_idx, neighborhoods, q, n, radius, time_limit=20.0):
    """Fast local search using precomputed neighborhoods."""
    t0 = time.time()
    N = len(all_words)
    code = [list(c) for c in code]
    improved = True
    
    while improved and (time.time() - t0) < time_limit:
        improved = False
        
        # Try removing each codeword
        for i in range(len(code)):
            if (time.time() - t0) > time_limit:
                break
            remaining = [c for j, c in enumerate(code) if j != i]
            # Check if remaining covers everything
            covered = np.zeros(N, dtype=bool)
            for c in remaining:
                idx = word_to_idx[tuple(c)]
                covered[neighborhoods[idx]] = True
            if covered.all():
                code = remaining
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved and (time.time() - t0) < time_limit:
            for i in range(len(code)):
                if (time.time() - t0) > time_limit:
                    break
                remaining = [c for j, c in enumerate(code) if j != i]
                # Find uncovered words with just 'remaining'
                covered_rem = np.zeros(N, dtype=bool)
                for c in remaining:
                    idx = word_to_idx[tuple(c)]
                    covered_rem[neighborhoods[idx]] = True
                
                uncovered = np.where(~covered_rem)[0]
                if len(uncovered) == 0:
                    code = remaining
                    improved = True
                    break
                
                # Try to find a single word that covers all uncovered
                for _ in range(100):
                    new_idx = random.choice(uncovered)
                    candidate = remaining + [list(all_words[new_idx])]
                    covered_cand = covered_rem.copy()
                    covered_cand[neighborhoods[new_idx]] = True
                    if covered_cand.all():
                        code = candidate
                        improved = True
                        break
                if improved:
                    break
    
    return code


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # Precompute neighborhoods
    all_words, word_to_idx, neighborhoods = _precompute_neighborhoods(q, n, radius)
    N = len(all_words)

    # Start with the known construction
    code = _base_tetracode_sum(q, n, radius)
    if code is None:
        code = []

    # Verify it's covering
    if not _is_covering(code, q, n, radius):
        # Fallback to pure greedy
        code = _greedy_cover_fast(all_words, word_to_idx, neighborhoods, q, n, radius, start_code=[], time_limit=60.0)

    # Try local search to improve (reduce) the code
    if time.time() < deadline:
        code = _local_search_improve_fast(code, all_words, word_to_idx, neighborhoods, q, n, radius, time_limit=min(40.0, deadline - time.time()))

    # Try greedy from scratch with a time limit and compare
    if time.time() < deadline:
        greedy_code = _greedy_cover_fast(all_words, word_to_idx, neighborhoods, q, n, radius, start_code=[], time_limit=min(40.0, deadline - time.time()))
        if greedy_code and len(greedy_code) < len(code):
            code = greedy_code
            # Try to improve the greedy result
            if time.time() < deadline:
                code = _local_search_improve_fast(code, all_words, word_to_idx, neighborhoods, q, n, radius, time_limit=min(20.0, deadline - time.time()))

    # Final check
    if not _is_covering(code, q, n, radius):
        # Emergency: return the base construction if it's valid
        base = _base_tetracode_sum(q, n, radius)
        if base is not None and _is_covering(base, q, n, radius):
            return base
        # Otherwise, do a quick greedy
        return _greedy_cover_fast(all_words, word_to_idx, neighborhoods, q, n, radius, start_code=[], time_limit=30.0)

    return code
# EVOLVE-BLOCK-END
