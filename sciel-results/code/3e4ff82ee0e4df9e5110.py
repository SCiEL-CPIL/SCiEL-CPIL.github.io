# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

# Deadline guard
_DEADLINE = 150.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def construct(q, n, radius):
    """
    Construct a covering code for the ternary Hamming space {0,1,2}^n
    with covering radius `radius`. Returns a list of codewords (each a list of n ints).
    """
    global _START_TIME
    _START_TIME = time.time()

    # For the specific task: q=3, n=7, radius=1
    # General fallback: use greedy set cover with precomputed spheres

    total_words = q ** n
    all_words = list(product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres: for each word, the set of words within Hamming distance <= radius
    # This is expensive for large n, but for n=7, q=3 it's manageable (2187 words)
    spheres = []
    for w in all_words:
        sphere = set()
        # Generate all words within Hamming distance <= radius
        # For radius=1: the word itself + all words differing in exactly one position
        if radius == 0:
            sphere = {w}
        elif radius == 1:
            sphere = {w}
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        sphere.add(tuple(new_w))
        else:
            # General case: use itertools to generate all combinations of positions to change
            # and all values to change to
            sphere = {w}
            # For radius r, we need to consider all subsets of positions of size <= r
            # and all possible value changes at those positions
            # This is expensive, so we'll use a simpler approach for small n
            for r in range(1, radius + 1):
                for positions in product(range(n), repeat=r):
                    # Skip if any position is repeated
                    if len(set(positions)) < r:
                        continue
                    for vals in product(range(q), repeat=r):
                        new_w = list(w)
                        for i, (pos, val) in enumerate(zip(positions, vals)):
                            if val != new_w[pos]:
                                new_w[pos] = val
                        sphere.add(tuple(new_w))
        spheres.append(sphere)

    # Convert spheres to numpy arrays for fast operations
    # We'll use a bitset approach: represent each sphere as a set of indices
    sphere_indices = []
    for sphere in spheres:
        idx_set = set(word_to_idx[w] for w in sphere)
        sphere_indices.append(idx_set)

    # Greedy set cover
    uncovered = set(range(total_words))
    codewords = []

    # Precompute sphere sizes for efficiency
    sphere_sizes = [len(s) for s in sphere_indices]

    # Sort indices by sphere size (descending) for initial ordering
    # But we'll do dynamic greedy: at each step, pick the word whose sphere covers the most uncovered words

    while uncovered and _time_left() > 5.0:
        # Find the word whose sphere covers the most uncovered words
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            if i not in uncovered and i not in [word_to_idx[tuple(cw)] for cw in codewords]:
                # We can still pick any word, not just uncovered ones
                pass
            count = len(sphere_indices[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1 or best_count == 0:
            # No improvement possible
            break

        codewords.append(list(all_words[best_idx]))
        uncovered -= sphere_indices[best_idx]

    # If we ran out of time or couldn't improve, return what we have
    if not codewords:
        # Fallback: return a simple construction
        # Direct product construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        else:
            return [list(range(n))]  # Degenerate fallback

    return codewords


def _greedy_construct(q, n, radius):
    """Alternative greedy construction with better initialization."""
    global _START_TIME
    _START_TIME = time.time()

    total_words = q ** n
    all_words = list(product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres
    spheres = []
    for w in all_words:
        sphere = set()
        if radius == 0:
            sphere = {w}
        elif radius == 1:
            sphere = {w}
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w = list(w)
                        new_w[pos] = val
                        sphere.add(tuple(new_w))
        else:
            sphere = {w}
            for r in range(1, radius + 1):
                for positions in product(range(n), repeat=r):
                    if len(set(positions)) < r:
                        continue
                    for vals in product(range(q), repeat=r):
                        new_w = list(w)
                        for i, (pos, val) in enumerate(zip(positions, vals)):
                            if val != new_w[pos]:
                                new_w[pos] = val
                        sphere.add(tuple(new_w))
        spheres.append(sphere)

    sphere_indices = [set(word_to_idx[w] for w in sphere) for sphere in spheres]

    uncovered = set(range(total_words))
    codewords = []

    # Use a priority-based approach: track for each candidate how many new words it covers
    # Start with a random seed or use the most central word
    if _time_left() < 5.0:
        return [list(all_words[0])]

    # Initial: pick the word with largest sphere (they're all same size for radius=1 in Hamming space)
    # So just pick word 0
    best_idx = 0
    codewords.append(list(all_words[0]))
    uncovered -= sphere_indices[0]

    while uncovered and _time_left() > 5.0:
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            count = len(sphere_indices[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        codewords.append(list(all_words[best_idx]))
        uncovered -= sphere_indices[best_idx]

    return codewords


def construct(q, n, radius):
    """
    Main entry point. Uses greedy set cover with precomputed spheres.
    """
    global _START_TIME
    _START_TIME = time.time()

    # For the specific case q=3, n=7, radius=1, use optimized greedy
    if q == 3 and n == 7 and radius == 1:
        return _greedy_construct(3, 7, 1)
    else:
        return _greedy_construct(q, n, radius)
# EVOLVE-BLOCK-END
