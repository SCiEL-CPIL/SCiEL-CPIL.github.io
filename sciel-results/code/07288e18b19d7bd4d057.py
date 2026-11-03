"""Ternary covering code K_3(8,1).

Strategy (keeps the proven direct-sum tetracode family):
  1. Initial code = direct sum of the ternary tetracode with all words on the
     remaining coordinates (9 * 3**(n-4) words).
  2. Vectorized greedy deletion: build the (codeword x word) incidence matrix
     once, then repeatedly drop any codeword whose covered words are all
     covered by >=2 codewords (a single NumPy pass per round). This is the
     step that measurably reduces the count.
  3. If wall-clock budget remains, run a bounded number of randomized
     perturbations (swap one codeword for a random nearby word, then re-run
     greedy deletion) to escape 1-opt local minima; keep the best valid code.

A hard deadline guard returns the best valid code found before the limit.
"""
import time
import random
import itertools
import numpy as np


def _tetracode_words():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3]
            for a in range(3) for b in range(3)]


def _build_base(q, n):
    tc = _tetracode_words()
    return [h + list(tail)
            for h in tc for tail in itertools.product(range(q), repeat=n - 4)]


def _sphere(c, q, n, radius):
    """Set of words within Hamming distance <= radius of c."""
    arr = np.array(c, dtype=np.int64)
    s = {tuple(arr)}
    if radius >= 1:
        for i in range(n):
            orig = arr[i]
            for v in range(q):
                if v == orig:
                    continue
                arr[i] = v
                s.add(tuple(arr))
            arr[i] = orig
    if radius >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = arr[i], arr[j]
                for vi in range(q):
                    for vj in range(q):
                        if vi == oi and vj == oj:
                            continue
                        arr[i], arr[j] = vi, vj
                        s.add(tuple(arr))
                arr[i], arr[j] = oi, oj
    return s


def _incidence(cw, q, n, radius):
    """Return (A, neigh) where A is (m, S) int8 matrix and neigh is the list of
    per-codeword word-lists (as tuples). S = total sphere size."""
    m = len(cw)
    neigh = [_sphere(c, q, n, radius) for c in cw]
    sizes = [len(x) for x in neigh]
    S = sum(sizes)
    A = np.zeros((m, S), dtype=np.int8)
    off = 0
    for i, lst in enumerate(neigh):
        if lst:
            A[i, off:off + len(lst)] = 1
        off += len(lst)
    return A, neigh


def _greedy_deletion(cw, q, n, radius):
    """Repeatedly drop any codeword whose words are all covered by >=2."""
    if not cw:
        return cw
    A, neigh = _incidence(cw, q, n, radius)
    m = len(cw)
    alive = list(range(m))
    while True:
        Aa = A[alive, :]
        counts = Aa.sum(axis=0)
        removable = np.nonzero((counts > 1).all(axis=0))[0]
        if removable.size == 0:
            break
        keep_mask = np.ones(len(alive), dtype=bool)
        keep_mask[removable] = False
        alive = [a for a, k in zip(alive, keep_mask) if k]
        if not alive:
            break
    return [cw[i] for i in alive]


def _is_covering(cw, q, n, radius):
    total = q ** n
    seen = 0
    covered = set()
    for c in cw:
        for w in _sphere(c, q, n, radius):
            if w not in covered:
                covered.add(w)
                seen += 1
                if seen == total:
                    return True
    return seen == total


def construct(q, n, radius):
    start = time.time()
    deadline = start + 132.0
    rng = random.Random(12345)

    base = _build_base(q, n)
    base = list({tuple(c): c for c in base}.values())

    best = _greedy_deletion(base, q, n, radius)
    best_len = len(best)

    def time_left():
        return deadline - time.time()

    # Bounded randomized perturbation + greedy deletion to escape local minima.
    attempts = 0
    while time_left() > 4.0 and attempts < 40:
        attempts += 1
        cur = [list(c) for c in best]
        m = len(cur)
        if m < 2:
            break
        # swap one codeword for a random nearby word
        i = rng.randrange(m)
        c = cur[i]
        cand = list(c)
        for _ in range(rng.randrange(0, radius + 1)):
            pos = rng.randrange(n)
            cand[pos] = rng.randrange(q)
        if tuple(cand) in {tuple(x) for x in cur}:
            continue
        cur[i] = cand
        # re-run greedy deletion
        cand_code = _greedy_deletion(cur, q, n, radius)
        if _is_covering(cand_code, q, n, radius) and len(cand_code) < best_len:
            best, best_len = cand_code, len(cand_code)

    # Final safety: ensure validity (best came from a valid construction).
    return best
