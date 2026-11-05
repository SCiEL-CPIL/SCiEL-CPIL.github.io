"""Football pool / ternary covering code K_3(7,1).

Strategy (a different design family from the direct-sum seed):
  * Greedy set-cover over the 3**n space: repeatedly add the codeword whose
    radius-r ball covers the most not-yet-covered words.
  * Vectorized NumPy distance/coverage computation so many greedy steps fit
    in the wall-clock budget.
  * Post-hoc redundancy pruning: drop any codeword that is not the unique
    cover of at least one word (i.e. whose ball is redundant).
  * Explicit deadline guard returns the best valid code before the limit.
"""
import time
import itertools
import numpy as np


def _ball_words(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of `word`."""
    word = list(word)
    idxs = range(n)
    for r in range(radius + 1):
        for pos in itertools.combinations(idxs, r):
            for vals in itertools.product(range(q), repeat=r):
                w = word[:]
                for p, v in zip(pos, vals):
                    w[p] = v
                yield tuple(w)


def _coverage_mask(words_arr, q, n, radius):
    """Boolean array of length q**n: word is covered if within radius of any
    word in words_arr."""
    total = q ** n
    if len(words_arr) == 0:
        return np.zeros(total, dtype=bool)
    allw = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    diff = (words_arr[:, None, :] != allw[None, :, :])
    dist = diff.sum(axis=2)
    return (dist <= radius).any(axis=0)


def _greedy_cover(q, n, radius, deadline):
    total = q ** n
    allw = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    covered = np.zeros(total, dtype=bool)
    chosen = []
    while not covered.all() and time.time() < deadline:
        diff = (allw[None, :, :] != allw[:, None, :])  # (total, total, n)
        dist = diff.sum(axis=2)
        can = dist <= radius
        gains = (can & ~covered[None, :]).sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] <= 0:  # safety: nothing left to cover
            break
        covered |= can[best]
        chosen.append(allw[best])
    return chosen


def _prune_redundant(code, q, n, radius):
    """Drop codewords that are not the unique cover of any word."""
    code = list(code)
    m = len(code)
    if m <= 1:
        return code
    total = q ** n
    allw = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    codew_arr = np.array(code, dtype=np.int8)
    diff = (allw[None, :, :] != codew_arr[:, None, :])  # (total, m, n)
    dist = diff.sum(axis=2)
    within = dist <= radius  # (total, m)
    counts = within.sum(axis=1)
    unique = np.where(counts == 1)[0]
    keep = np.unique(within[unique])
    return [code[i] for i in keep]


def construct(q, n, radius):
    t0 = time.time()
    # Safety margin: hard evaluator limit is 150 s wall-clock.
    deadline = t0 + 140.0

    # Greedy set-cover construction.
    code = _greedy_cover(q, n, radius, deadline)

    # Redundancy pruning (cheap, vectorized).
    if time.time() < deadline:
        code = _prune_redundant(code, q, n, radius)

    # Return distinct codewords as lists of ints.
    seen = set()
    out = []
    for w in code:
        t = tuple(int(x) for x in w)
        if t not in seen:
            seen.add(t)
            out.append(list(t))
    return out
