import time
import numpy as np
from itertools import combinations
import random

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0


def _seed_words_rm():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words


def _build_forbidden_mask(words):
    """Build a boolean mask of size 2^N where True means 'forbidden'."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden


def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results


def _local_search(seed_words, deadline):
    """
    Perform local search to expand the code.
    Strategy: greedy insertion with multiple randomized passes.
    """
    words = list(seed_words)
    forbidden = _build_forbidden_mask(words)
    
    # Create a pool of candidates (all words except seed)
    seed_set = set(words)
    candidates = [w for w in range(TOTAL) if w not in seed_set]
    
    # Multiple passes with different shuffles
    best_words = list(words)
    best_forbidden = forbidden.copy()
    
    for pass_num in range(5):
        if time.time() > deadline:
            break
        
        # Try a fresh shuffle each time
        random.shuffle(candidates)
        current_words = list(words)
        current_forbidden = forbidden.copy()
        
        for w in candidates:
            if time.time() > deadline:
                break
            if not current_forbidden[w]:
                current_words.append(w)
                neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
                current_forbidden[neighbors] = True
        
        if len(current_words) > len(best_words):
            best_words = current_words
            best_forbidden = current_forbidden
            words = current_words
            forbidden = current_forbidden
    
    return best_words


def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    # Seed with Reed-Muller code
    seed = _seed_words_rm()
    
    # Run local search (which includes greedy expansion)
    try:
        w = _local_search(seed, deadline)
        if len(w) > best_len:
            best_len = len(w)
            best_words = w
    except Exception:
        pass

    if best_words is None:
        best_words = seed

    # Deduplicate and validate
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    if not out:
        out = seed

    return out
