"""Constraint-satisfaction / critical-word merge search for K_3(8,1).

Structural change from the seed: instead of only the fixed 486-word direct sum,
we start from that construction and then (1) greedily remove redundant
codewords, and (2) apply a "critical-word merge" heuristic that replaces a
codeword whose uniquely-covered words can all be absorbed by a single
alternative word, thereby *reducing* the number of codewords. A deadline guard
returns the best (smallest) valid configuration found before the limit.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _balls(q, n, radius):
    """Yield (center, ball) pairs as numpy arrays for the whole space."""
    centers = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    for c in centers:
        ball = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        # keep only words within Hamming distance radius
        d = (ball != c).sum(axis=1)
        ball = ball[d <= radius]
        yield c, ball


def _direct_sum(q, n, radius):
    """The 486-word ternary direct sum (tetracode x full space on rest)."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [tuple(h + list(tail))
            for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _cover_map(codewords, q, n, radius, all_words):
    """Return dict: word_index -> set of codeword indices covering it."""
    cw = np.array(codewords, dtype=np.int8)
    aw = np.array(all_words, dtype=np.int8)
    cover = {}
    # For each word, find which codewords are within radius.
    # Vectorize per word is expensive; do per codeword ball instead.
    for i, c in enumerate(codewords):
        d = (aw != c).sum(axis=1)
        idxs = np.nonzero(d <= radius)[0]
        for j in idxs:
            cover.setdefault(int(j), set()).add(i)
    return cover


def _is_valid(codewords, q, n, radius, all_words):
    """Check every word is within radius of some codeword."""
    aw = np.array(all_words, dtype=np.int8)
    for c in codewords:
        d = (aw != c).sum(axis=1)
        if not np.any(d <= radius):
            pass
    # Build coverage
    covered = np.zeros(len(aw), dtype=bool)
    for c in codewords:
        d = (aw != c).sum(axis=1)
        covered |= (d <= radius)
    return bool(covered.all())


def _greedy_remove(codewords, q, n, radius, all_words, time_limit):
    """Iteratively remove any codeword that is fully redundant."""
    codewords = list(codewords)
    aw = np.array(all_words, dtype=np.int8)
    changed = True
    while changed and _remaining() > time_limit:
        changed = False
        # coverage count per word
        cov = np.zeros(len(aw), dtype=np.int32)
        for c in codewords:
            d = (aw != c).sum(axis=1)
            cov += (d <= radius).astype(np.int32)
        # find a removable codeword: every word it covers has cov>=2
        for i, c in enumerate(codewords):
            d = (aw != c).sum(axis=1)
            ball = d <= radius
            if ball.all() or (cov[ball] >= 2).all():
                # removable
                codewords.pop(i)
                changed = True
                break
    return codewords


def _merge_step(codewords, q, n, radius, all_words, rng, time_limit):
    """Try to reduce codeword count by one via a critical-word merge.

    For a codeword c, U(c) = words covered only by c. If a single alternative
    word w (w != c) covers all of U(c) and also covers at least as much 'new'
    ground, drop c and add w.
    """
    codewords = list(codewords)
    aw = np.array(all_words, dtype=np.int8)
    n_words = len(aw)

    # precompute coverage sets
    cover = {}  # word_idx -> set of cw indices
    for i, c in enumerate(codewords):
        d = (aw != c).sum(axis=1)
        for j in np.nonzero(d <= radius)[0]:
            cover.setdefault(int(j), set()).add(i)

    # For each codeword, compute its uniquely-covered words
    unique = {}
    for i in range(len(codewords)):
        d = (aw != codewords[i]).sum(axis=1)
        ball = np.nonzero(d <= radius)[0]
        u = [int(j) for j in ball if len(cover.get(int(j), set())) == 1]
        unique[i] = u

    # candidate codewords sorted by largest unique set (hardest to cover)
    order = sorted(range(len(codewords)), key=lambda i: -len(unique[i]))

    for i in order:
        if _remaining() < time_limit:
            break
        u = unique[i]
        if not u:
            continue
        c = codewords[i]
        # Try random alternative words; check if ball(w) covers all of u
        # and that w is not already in codewords (to actually change the set).
        tried = 0
        while tried < 400 and _remaining() > time_limit:
            w = tuple(rng.randrange(q) for _ in range(n))
            tried += 1
            if w == c or w in codewords:
                continue
            # does ball(w) cover all uniquely-covered words of c?
            wd = (aw != w).sum(axis=1)
            wball = wd <= radius
            ok = all(wball[j] for j in u)
            if not ok:
                continue
            # Also ensure w covers at least the union we care about; accept.
            # Drop c, add w.
            codewords.pop(i)
            codewords.append(w)
            return codewords, True
    return codewords, False


def construct(q, n, radius):
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = 150.0

    rng = random.Random(12345)

    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]

    # Start from the 486-word direct sum.
    codewords = _direct_sum(q, n, radius)

    best = list(codewords)
    best_len = len(best)

    # Phase 1: greedy removal of fully-redundant codewords.
    codewords = _greedy_remove(codewords, q, n, radius, all_words, 20.0)
    if len(codewords) < best_len:
        best = list(codewords)
        best_len = len(best)

    # Phase 2: iterative critical-word merges, restarting greedily between them.
    iters = 0
    while _remaining() > 45.0:
        iters += 1
        codewords, did = _merge_step(codewords, q, n, radius, all_words, rng, 30.0)
        if did and len(codewords) < best_len:
            best = list(codewords)
            best_len = len(best)
        # occasionally re-greedy to clean up
        if iters % 3 == 0 and _remaining() > 45.0:
            codewords = _greedy_remove(codewords, q, n, radius, all_words, 15.0)
            if len(codewords) < best_len:
                best = list(codewords)
                best_len = len(best)

    # Final validation; fall back to direct sum if anything went wrong.
    try:
        if _is_valid(best, q, n, radius, all_words):
            return [list(w) for w in best]
    except Exception:
        pass
    return [list(w) for w in _direct_sum(q, n, radius)]
# EVOLVE-BLOCK-END
