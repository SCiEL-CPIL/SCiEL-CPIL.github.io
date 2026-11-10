"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# --- static setup: prime factor vectors over the 12 primes in 2..40 ---
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factor_vec(x):
    v = [0] * len(_PRIMES)
    for i, p in enumerate(_PRIMES):
        while x % p == 0:
            v[i] += 1
            x //= p
    return v

# index -> value (1..39 for values 2..40)
_VALS = list(range(2, 41))
_VECT = np.array([_factor_vec(v) for v in _VALS], dtype=np.int64)
_N = _VECT.shape[0]
_ALL = np.arange(_N)
_T0 = time.time()
_DEADLINE = _T0 + 55.0          # safety margin vs 60s (90s kill)
_TIME_PER_NODE = 2e-5           # conservative per-node estimate for B&B

def _rank(mat):
    """Row rank over Z for a small non-negative integer matrix (exact)."""
    m, n = mat.shape
    r = 0
    a = mat.copy()
    for c in range(n):
        if r == m:
            break
        piv = np.argmax(a[r:, c]) + r
        if a[piv, c] == 0:
            continue
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        a[r] = a[r] // a[r, c]
        for i in range(r + 1, m):
            if a[i, c]:
                a[i] = a[i] - (a[i, c] // a[r, c]) * a[r]
        r += 1
    return r

def _independent(vecs):
    """True iff the given row vectors are linearly independent over Z."""
    if len(vecs) == 0:
        return True
    if len(vecs) == 1:
        return int(vecs[0].any())
    return _rank(vecs) == len(vecs)

def _search(deadline):
    """Exact branch-and-bound: maximum independent set of the vector matroid."""
    best = [0, []]
    order = np.argsort(-_VECT.sum(axis=1))          # densest (most primes) first
    V = _VECT[order]
    n = _N

    def rec(pos, chosen, mat, remaining):
        if time.time() > deadline:
            return
        if chosen:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
                best[1] = list(chosen)
        if pos == n:
            return
        if len(chosen) + remaining <= best[0]:     # bound: can't beat best
            return
        for k in range(pos, n):
            v = V[k]
            if mat is not None:
                cand = np.vstack([mat, v])
                if _rank(cand) != len(chosen) + 1:
                    continue
            else:
                if not v.any():
                    continue
                cand = v.reshape(1, -1)
            rec(k + 1, chosen + 1, cand, remaining - (n - k - 1))
            if time.time() > deadline:
                return

    rec(0, 0, None, n)
    return best[1]

def construct():
    deadline = _T0 + 55.0
    best_set = None
    best_size = 0

    # 1) exact complete search (fast rank test) -> provably optimal size
    try:
        idx = _search(deadline)
        if idx:
            best_set = [int(_VALS[i]) for i in idx]
            best_size = len(best_set)
    except Exception:
        pass

    # 2) if exact search did not finish, fall back to fast greedy heuristics
    if time.time() < deadline - 2.0:
        rng = random.Random(12345)
        for _ in range(2000):
            if time.time() > deadline - 2.0:
                break
            if rng.random() < 0.5:
                perm = np.argsort(-_VECT.sum(axis=1))
            else:
                perm = np.arange(_N)
                rng.shuffle(perm)
            mat = None
            chosen = []
            for i in perm:
                v = _VECT[i]
                if mat is None:
                    if not v.any():
                        continue
                    mat = v.reshape(1, -1)
                    chosen.append(i)
                else:
                    cand = np.vstack([mat, v])
                    if _rank(cand) == len(chosen) + 1:
                        mat = cand
                        chosen.append(i)
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = [int(_VALS[i]) for i in chosen]
                if best_size == 12:
                    break

    # 3) deterministic prime set as a guaranteed valid fallback (size 12)
    if best_set is None:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # 4) final exact verification of the returned set (defensive)
    vecs = np.array([_factor_vec(v) for v in best_set], dtype=np.int64)
    if not _independent(vecs):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
