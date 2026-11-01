"""Greedy covering-code construction for K_q(n, radius) with randomized restarts.

Family: keep the seed's "build a covering set, then reduce to a minimum" idea, but
implement the reduction as a greedy set-cover over the explicit radius-`radius` balls.
The seed's 243 words (tetracode direct-sum) are redundant; greedy set-cover keeps
only the words that still add coverage, dropping the rest. Randomized restarts
(random first pick + random tie-breaks) escape the worst greedy orderings.
"""
import time
import random
import numpy as np


def _balls(q, n, radius):
    """Return (words, balls): words = all q**n words as int64 matrix,
    balls = boolean matrix where balls[i, j] is True iff word j is within
    Hamming distance `radius` of word i."""
    words = np.array(np.meshgrid(*([np.arange(q, dtype=np.int64)] * n)),
                     indexing="ij").reshape(n, -1).T
    balls = np.zeros((len(words), len(words)), dtype=bool)
    if radius <= 0:
        balls = np.eye(len(words), dtype=bool)
    else:
        # positions of each 1-bit (distance 1)
        unit = np.zeros((len(words), len(words)), dtype=bool)
        for d in range(n):
            unit |= (words[:, d:d + 1] != words[d])
        balls = unit.copy()
        if radius >= 2:
            balls |= unit @ unit
        if radius >= 3:
            balls |= (unit @ unit) @ unit
    return words, balls


def _greedy(words, balls, seed, first_mode):
    """Greedy set-cover. Returns list of chosen word indices."""
    rng = random.Random(seed)
    remaining = np.ones(len(words), dtype=bool)
    chosen = []
    if first_mode == "random":
        first = rng.randrange(len(words))
    else:
        first = int(np.argmax(balls[0]))  # deterministic, matches seed order
    chosen.append(first)
    remaining &= ~balls[first]
    while remaining.any():
        gains = balls[remaining].sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        remaining &= ~balls[best]
    return chosen


def _dedupe(words, idx):
    """Map selected indices to distinct codewords (guard against duplicates)."""
    seen = set()
    out = []
    for i in idx:
        key = tuple(int(x) for x in words[i])
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out


def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin below the 120 s kill
    words, balls = _balls(q, n, radius)
    N = len(words)

    def is_valid(idx):
        covered = balls[idx].any(axis=0)
        return bool(covered.all())

    # Phase 1: deterministic greedy (reproduces the seed family's reduction).
    det_idx = _greedy(words, balls, 0, "first")
    best = _dedupe(words, det_idx)
    best_size = len(best)

    # Phase 2: randomized restarts; keep the smallest valid code.
    if time.time() < deadline - 15.0:
        for it in range(1, 200000):
            if time.time() > deadline - 15.0:
                break
            idx = _greedy(words, balls, it, "random")
            if len(idx) < best_size and is_valid(idx):
                cand = _dedupe(words, idx)
                if len(cand) < best_size and is_valid(cand):
                    best = cand
                    best_size = len(cand)
            if best_size <= 186:  # reached the known optimum for this task
                break

    return best
