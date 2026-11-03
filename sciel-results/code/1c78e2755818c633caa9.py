"""Ternary covering code K_3(7,1) via greedy + local search."""
import itertools
import time
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _all_neighbors(word, q, n, radius=1):
    """Yield all words within Hamming distance <= radius of word."""
    yield tuple(word)
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                yield tuple(w)


def _covers(codewords, q, n, radius=1):
    """Return a set of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        for w in _all_neighbors(cw, q, n, radius):
            covered.add(w)
    return covered


def _greedy_cover(q, n, radius=1, seed=42):
    """Greedy set-cover: repeatedly pick the codeword covering the most uncovered words."""
    rng = random.Random(seed)
    total = q ** n
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []

    # Precompute coverage for each candidate
    # For each word, compute what it covers
    coverage = {}
    for w in all_words:
        cov = set()
        for v in _all_neighbors(w, q, n, radius):
            cov.add(v)
        coverage[w] = cov

    while uncovered:
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        # Sample if too many candidates to check all
        if len(uncovered) > 500:
            candidates = list(uncovered)
            rng.shuffle(candidates)
            candidates = candidates[:500]
        else:
            candidates = list(uncovered)

        for w in candidates:
            cnt = len(coverage[w] & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_word = w

        if best_word is None or best_count == 0:
            break

        codewords.append(list(best_word))
        uncovered -= coverage[best_word]

    return codewords


def _try_remove(codewords, q, n, radius=1, idx=None):
    """Try to remove codeword at idx (or a random one) and check if still covered."""
    if len(codewords) <= 1:
        return codewords, False

    if idx is None:
        idx = random.randint(0, len(codewords) - 1)

    removed = codewords[idx]
    remaining = [c for i, c in enumerate(codewords) if i != idx]

    # Check if still covered
    covered = _covers(remaining, q, n, radius)
    total = q ** n
    if len(covered) == total:
        return remaining, True
    return codewords, False


def _local_search(codewords, q, n, radius=1, max_rounds=50, seed=42):
    """Iteratively try to remove codewords until none can be removed."""
    rng = random.Random(seed)
    codewords = list(codewords)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        # Try removing each codeword
        order = list(range(len(codewords)))
        rng.shuffle(order)
        for idx in order:
            new_code, removed = _try_remove(codewords, q, n, radius, idx)
            if removed:
                codewords = new_code
                improved = True
                # After removal, indices shifted, restart
                break
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Initial greedy cover
    codewords = _greedy_cover(q, n, radius, seed=42)
    best = list(codewords)

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best

    # Local search to reduce
    codewords = _local_search(codewords, q, n, radius, max_rounds=100, seed=42)
    if len(codewords) < len(best):
        best = list(codewords)

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best

    # Try multiple greedy seeds and keep the best
    for seed in [1, 7, 13, 99, 123, 456, 789, 1000, 2024, 3141]:
        if time.time() - start_time > deadline:
            break
        cw = _greedy_cover(q, n, radius, seed=seed)
        cw = _local_search(cw, q, n, radius, max_rounds=50, seed=seed)
        if len(cw) < len(best):
            best = list(cw)

    return best
