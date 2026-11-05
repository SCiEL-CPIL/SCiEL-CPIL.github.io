"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

_START_TIME = None
_DEADLINE_FRAC = 0.85  # safety margin: stop at 85% of 120s


def _elapsed():
    return time.time() - _START_TIME


def _within_deadline(margin=0.0):
    if _START_TIME is None:
        return True
    limit = 120.0 * _DEADLINE_FRAC
    return (time.time() - _START_TIME) < (limit - margin)


def _tetracode_sum(q, n, radius):
    """Direct sum of ternary tetracode with all tails: 9 * 3**(n-4) codewords."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_indices(codeword, q, n, radius):
    """Yield the flat index of every word within Hamming distance <= radius of codeword."""
    codeword = tuple(codeword)
    for r in range(radius + 1):
        for positions in itertools.combinations(range(n), r):
            for vals in itertools.product(range(1, q), repeat=r):
                w = list(codeword)
                for pos, val in zip(positions, vals):
                    w[pos] = (codeword[pos] + val) % q
                yield _flat(w, q, n)


def _flat(word, q, n):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _uncovered_mask(codewords, q, n, radius):
    """Return a boolean numpy array of size q**n; True = covered."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        for idx in _ball_indices(cw, q, n, radius):
            covered[idx] = True
    return covered


def _is_valid(codewords, q, n, radius):
    covered = _uncovered_mask(codewords, q, n, radius)
    return covered.all()


def _greedy_construct(q, n, radius):
    """Greedy set-cover: repeatedly pick the codeword covering the most uncovered words."""
    total = q ** n
    # Precompute all possible codewords and their balls
    all_cw = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_balls = [set(_ball_indices(cw, q, n, radius)) for cw in all_cw]
    
    uncovered = set(range(total))
    codewords = []
    
    while uncovered and _within_deadline():
        best_i = -1
        best_count = 0
        for i, ball in enumerate(all_balls):
            if i in codewords:
                continue
            count = len(ball & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
            if best_count == len(uncovered):
                break
        if best_i < 0 or best_count == 0:
            break
        codewords.append(all_cw[best_i])
        uncovered -= all_balls[best_i]
    
    return codewords


def _local_search(codewords, q, n, radius, max_iters=500):
    """Local search: try removing redundant codewords and swapping pairs."""
    codewords = [list(cw) for cw in codewords]
    total = q ** n
    
    # Precompute balls for current codewords
    balls = [set(_ball_indices(cw, q, n, radius)) for cw in codewords]
    
    # Compute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1
    
    improved = True
    iters = 0
    while improved and iters < max_iters and _within_deadline():
        improved = False
        iters += 1
        
        # Phase 1: Try removing each codeword if it doesn't break coverage
        for i in range(len(codewords) - 1, -1, -1):
            if not _within_deadline():
                break
            # Check if removing codeword i leaves all words covered
            can_remove = True
            for idx in balls[i]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                for idx in balls[i]:
                    coverage[idx] -= 1
                del codewords[i]
                del balls[i]
                improved = True
                break  # restart after removal
        
        # Phase 2: Try swapping one codeword with another (add one, remove one)
        if not improved and len(codewords) > 1:
            # For each codeword, see if there's a replacement that allows removal
            # This is expensive, so we do a limited search
            for i in range(len(codewords)):
                if not _within_deadline():
                    break
                # Try adding a random codeword and then removing i
                # But first, let's just try to find if any single addition enables a removal
                pass  # Skip complex swap for now; focus on removals
    
    return codewords


def _greedy_then_local(q, n, radius):
    """Start from tetracode sum, then apply local search."""
    # Start from the 81-word tetracode sum
    codewords = _tetracode_sum(q, n, radius)
    
    # Apply local search to reduce
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def _greedy_with_removal(q, n, radius):
    """Greedy construction followed by removal-based local search."""
    # First, build a greedy solution
    codewords = _greedy_construct(q, n, radius)
    
    # Then apply local search
    codewords = _local_search(codewords, q, n, radius)
    
    return codewords


def construct(q, n, radius):
    global _START_TIME
    _START_TIME = time.time()
    
    # For the specific case q=3, n=6, radius=1, use the known structure
    if q == 3 and n == 6 and radius == 1:
        # Start from tetracode sum (81 words) and try to reduce to 73
        best = _tetracode_sum(q, n, radius)
        
        # Try local search starting from tetracode sum
        candidate = _local_search(best, q, n, radius)
        if len(candidate) < len(best):
            best = candidate
        
        # Try greedy construction + local search
        candidate = _greedy_with_removal(q, n, radius)
        if len(candidate) < len(best):
            best = candidate
        
        # Verify validity
        if _is_valid(best, q, n, radius):
            return best
        
        # Fallback: return the tetracode sum (known valid)
        return _tetracode_sum(q, n, radius)
    else:
        # General case: use greedy construction
        codewords = _greedy_construct(q, n, radius)
        if _is_valid(codewords, q, n, radius):
            return codewords
        # Fallback: brute-force all words
        return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
