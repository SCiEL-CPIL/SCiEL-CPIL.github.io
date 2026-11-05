"""Optimized constructor: minimize max(|A+A|, |A*A|) for 20 distinct ints <= 10000."""
import time
import random
from itertools import combinations_with_replacement

# EVOLVE-BLOCK-START
_N = 20
_MAX = 10000

def _sumset_size(A):
    # A sorted list; include equal pairs
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _score(A):
    return max(_sumset_size(A), _prodset_size(A))

def _build_smooth_pool():
    # numbers of form 2^a 3^b 5^c <= _MAX (5-smooth), distinct, >=1
    pool = set()
    a = 1
    while a <= _MAX:
        b = a
        while b <= _MAX:
            c = b
            while c <= _MAX:
                pool.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(x for x in pool if 1 <= x <= _MAX)

_POOL = _build_smooth_pool()

def _initial_A():
    # pick 20 spread smooth numbers, avoid 1 to reduce duplicate products
    # use a deterministic base then shuffle a bit
    cand = [x for x in _POOL if x >= 2]
    random.shuffle(cand)
    A = sorted(cand[:_N])
    return A

def _swap(A, idx, new_val):
    A2 = list(A)
    A2[idx] = new_val
    A2 = sorted(A2)
    return A2

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    rng = random.Random(12345)

    best_A = _initial_A()
    best_score = _score(best_A)

    # local search with random restarts
    while time.time() < deadline:
        A = list(best_A)
        # occasional full restart to escape local minima
        if rng.random() < 0.15:
            A = _initial_A()
        cur = _score(A)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(A)
            # candidate replacement pool: smooth numbers not in A
            present = set(A)
            cands = [x for x in _POOL if x not in present and 1 <= x <= _MAX]
            rng.shuffle(cands)
            # limit candidate count per step for speed
            cands = cands[:400]

            for idx in range(n):
                old = A[idx]
                for nv in cands:
                    if nv == old:
                        continue
                    A2 = _swap(A, idx, nv)
                    s = _score(A2)
                    if s < cur:
                        A = A2
                        cur = s
                        improved = True
                        break
                if improved:
                    break

        if cur < best_score:
            best_score = cur
            best_A = list(A)

    # final safety: ensure validity
    best_A = sorted(set(best_A))
    assert len(best_A) == _N
    assert all(1 <= x <= _MAX for x in best_A)
    return best_A
# EVOLVE-BLOCK-END
