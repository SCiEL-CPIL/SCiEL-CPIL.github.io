"""Rewritten program for the suite task; stronger algorithm for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import deque

import numpy as np


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _covers(codeword, word, radius):
    return _hamming_dist(codeword, word) <= radius


def _is_valid_cover(codewords, q, n, radius, full_check=True):
    """Check that every word in {0..q-1}^n is within radius of some codeword."""
    if not codewords:
        return n == 0
    # For small spaces, do an exhaustive check.
    total = q ** n
    if total <= 2000000:
        for word in itertools.product(range(q), repeat=n):
            if not any(_covers(cw, word, radius) for cw in codewords):
                return False
        return True
    # For larger spaces, we can't check everything; rely on construction guarantees
    # or spot-check. For our constructions we know they are valid.
    # But to be safe, we do a random sample if full_check is True.
    if full_check:
        rng = np.random.default_rng(42)
        for _ in range(min(100000, total)):
            word = tuple(int(x) for x in rng.integers(0, q, size=n))
            if not any(_covers(cw, word, radius) for cw in codewords):
                return False
    return True


def _greedy_cover(q, n, radius, max_iters=50000, deadline=None):
    """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
    # For efficiency, we work with the full space if small, else use a heuristic.
    total = q ** n
    if total > 500000:
        # Use a different approach for large spaces
        return _greedy_large_space(q, n, radius, max_iters, deadline)
    
    # Represent all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []
    
    # Precompute coverage for each possible codeword
    # A codeword is any word in the space
    # For each codeword, find which words it covers (within Hamming distance radius)
    
    # To speed up, we can use the fact that words within Hamming distance r from a codeword
    # are those that differ in at most r positions.
    
    # For small n and q, enumerate all possible codewords and their coverage
    # But this might be expensive. Let's use a smarter approach.
    
    # Actually, for q=3, n=7, radius=1, the space is 2187.
    # Each codeword covers 1 + 7*2 = 15 words (itself + 7 positions * 2 alternatives).
    # So we need at least ceil(2187/15) = 146 codewords.
    
    # Let's use a greedy approach where we pick codewords that maximize new coverage.
    
    # To make this efficient, we'll track uncovered words and for each candidate codeword,
    # count how many uncovered words it covers.
    
    # Candidate codewords are all words in the space.
    candidates = all_words[:]
    
    while uncovered and (deadline is None or time.time() < deadline):
        best_cw = None
        best_new_cover = 0
        
        for cw in candidates:
            # Count how many uncovered words are within radius of cw
            new_cover = 0
            for w in uncovered:
                if _hamming_dist(cw, w) <= radius:
                    new_cover += 1
            if new_cover > best_new_cover:
                best_new_cover = new_cover
                best_cw = cw
            if new_cover == len(uncovered):
                break
        
        if best_cw is None or best_new_cover == 0:
            break
        
        codewords.append(best_cw)
        candidates.remove(best_cw)
        # Remove covered words from uncovered
        to_remove = []
        for w in uncovered:
            if _hamming_dist(best_cw, w) <= radius:
                to_remove.append(w)
        for w in to_remove:
            uncovered.discard(w)
    
    return codewords


def _greedy_large_space(q, n, radius, max_iters=50000, deadline=None):
    """Greedy for large spaces using a different strategy."""
    # Use the tetracode-based construction as a starting point
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return codewords
    
    # Fallback: random greedy
    rng = np.random.default_rng(42)
    codewords = []
    # Just pick random codewords until we think we have enough
    # This is a weak fallback
    count = int(np.ceil(q ** n / (1 + n * (q - 1) * radius)))
    for _ in range(min(count + 100, 5000)):
        cw = [int(x) for x in rng.integers(0, q, size=n)]
        codewords.append(cw)
    return codewords


def _local_search(codewords, q, n, radius, deadline=None):
    """Try to reduce the code size by removing redundant codewords and adding better ones."""
    codewords = [list(cw) for cw in codewords]
    
    # First, try to remove redundant codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords)):
            # Check if codeword i is redundant
            # i.e., all words it covers are also covered by other codewords
            # For efficiency, check if removing it still leaves a valid cover
            # This is expensive, so only do it for small codes
            if len(codewords) <= 300:
                remaining = [cw for j, cw in enumerate(codewords) if j != i]
                if _is_valid_cover(remaining, q, n, radius, full_check=False):
                    codewords.pop(i)
                    changed = True
                    break
    
    return codewords


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin before 120s limit
    
    # Special case: perfect codes
    if q == 3 and n == 4 and radius == 1:
        # Ternary tetracode: 9 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return tetracode
    
    if q == 3 and n == 1 and radius == 1:
        return [[0], [1], [2]]
    
    if q == 2 and n == 3 and radius == 1:
        # Hamming(3,1) code: 4 words
        return [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    
    # General approach:
    # 1. Start with a good construction
    # 2. Try to improve via local search
    
    # Start with the tetracode-based construction if applicable
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Use greedy
        codewords = _greedy_cover(q, n, radius, deadline=deadline)
    
    # Try to improve with local search
    if time.time() < deadline:
        codewords = _local_search(codewords, q, n, radius, deadline=deadline)
    
    # Verify validity (spot check)
    if not _is_valid_cover(codewords, q, n, radius, full_check=True):
        # Fallback to a known valid construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Worst case: all words
            codewords = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    return codewords
# EVOLVE-BLOCK-END
