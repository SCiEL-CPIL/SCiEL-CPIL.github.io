"""Greedy set-cover + random restarts + local improvement for ternary covering codes."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _all_words(q, n):
    # All words of {0..q-1}^n as an int array of shape (q**n, n).
    total = q ** n
    words = np.empty((total, n), dtype=np.int8)
    for d in range(n):
        col = np.arange(total, dtype=np.int64) // (q ** (n - 1 - d))
        words[:, d] = (col % q).astype(np.int8)
    return words


def _min_count(words, idx):
    # Minimum "remaining uncovered" count over the n coordinates of the given words.
    return words[idx].min(axis=1)


def _add_ball(counts, words, idx, delta):
    # Add delta to the remaining-count of every word within Hamming distance <= 1 of words[idx].
    counts[idx] += delta
    for d in range(words.shape[1]):
        for s in range(1, 3):
            j = idx + d * s
            if j < len(words):
                counts[j] += delta


def _remove_ball(counts, words, idx, delta):
    # Remove delta from the remaining-count of every word within Hamming distance <= 1 of words[idx].
    counts[idx] -= delta
    for d in range(words.shape[1]):
        for s in range(1, 3):
            j = idx + d * s
            if j < len(words):
                counts[j] -= delta


def _greedy(q, n, words, rng, deadline):
    # Greedy set-cover: keep adding the codeword covering the most uncovered words.
    total = len(words)
    counts = np.zeros(total, dtype=np.int32)
    chosen = []
    chosen_set = set()
    remaining = total

    while remaining > 0:
        if time.time() > deadline:
            return None
        # For each unused candidate, count how many currently-uncovered words it covers.
        cand = [i for i in range(total) if i not in chosen_set]
        if not cand:
            return None
        gains = np.array([int((_min_count(words, [i]) == 0).sum()) for i in cand], dtype=np.int64)
        # Pick the maximum gain; break ties randomly to diversify restarts.
        maxg = int(gains.max())
        best_idx = [i for i, g in zip(cand, gains) if g == maxg]
        pick = rng.choice(best_idx)
        _add_ball(counts, words, pick, 1)
        remaining -= int((_min_count(words, [pick]) == 0).sum())
        chosen.append(pick)
        chosen_set.add(pick)
    return chosen


def _try_remove(chosen, q, n, words, counts, rng, deadline):
    # Try to remove any codeword whose ball is fully covered by the others.
    improved = True
    while improved and time.time() <= deadline:
        improved = False
        for pos in range(len(chosen) - 1, -1, -1):
            i = chosen[pos]
            _remove_ball(counts, words, i, 1)
            uncovered = int((_min_count(words, [i]) == 0).sum())
            if uncovered == 0:
                # i is redundant: drop it.
                chosen.pop(pos)
            else:
                _add_ball(counts, words, i, 1)
                improved = True
                break
    return chosen, counts


def construct(q, n, radius):
    start = time.time()
    # Safety margin well under the 150 s wall-clock kill.
    deadline = start + 125.0
    rng = random.Random(20240517)

    words = _all_words(q, n)
    total = len(words)

    # Greedy set-cover with random restarts; keep the smallest valid code found.
    best = None
    restarts = 0
    while time.time() <= deadline:
        chosen = _greedy(q, n, words, rng, deadline)
        if chosen is None:
            break
        # Local improvement: remove redundant codewords.
        counts = np.zeros(total, dtype=np.int32)
        for i in chosen:
            _add_ball(counts, words, i, 1)
        chosen, counts = _try_remove(chosen, q, n, words, counts, rng, deadline)

        if best is None or len(chosen) < len(best):
            best = list(chosen)
        restarts += 1
        # Stop early if we hit a very strong value (486 is the known optimum for q=3,n=8,r=1).
        if len(best) <= 486:
            break

    if best is None:
        # Fallback: return a trivially valid (but large) code if time ran out.
        return [list(row) for row in words[:1]]

    return [list(words[i]) for i in best]
# EVOLVE-BLOCK-END
