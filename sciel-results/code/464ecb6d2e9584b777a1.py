"""Rewritten program: vectorized simulated-annealing local search for R(4,5;3) coloring."""
import time
import random
import numpy as np
from itertools import combinations

# Hard wall-clock limit (evaluator kills at 150s; keep safety margin).
LIMIT = 120.0
DEADLINE = time.time() + LIMIT

# ---------- precomputed combinatorial structure (n <= 40) ----------
def _build(n):
    triples = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    M = triples.shape[0]
    A = np.zeros((M, n), dtype=np.int8)
    A[np.arange(M), triples[:, 0]] = 1
    A[np.arange(M), triples[:, 1]] = 1
    A[np.arange(M), triples[:, 2]] = 1

    # red K4: each 4-set -> 4 triples
    red4 = []
    for c in combinations(range(n), 4):
        idx = [i for i in range(M)
               if triples[i, 0] == c[0] and triples[i, 1] == c[1] and triples[i, 2] == c[2]]
        red4.append(idx)
    # blue K5: each 5-set -> 10 triples
    blue5 = []
    for c in combinations(range(n), 5):
        idx = [i for i in range(M)
               if triples[i, 0] == c[0] and triples[i, 1] == c[1] and triples[i, 2] == c[2]]
        blue5.append(idx)
    return triples, A, red4, blue5

def _count_violations(A, red4, blue5, x):
    # red: triples with color 0 ; blue: triples with color 1
    red = (x == 0)
    blue = (x == 1)
    v = 0
    for idx in red4:
        if red[idx].all():
            v += 1
    for idx in blue5:
        if blue[idx].all():
            v += 1
    return v

def _init(n, A, red4, blue5, rng):
    # random start, then a few cheap greedy flips to reduce violations
    x = (rng.random(A.shape[0]) < 0.5).astype(np.int8)
    for _ in range(200):
        v = _count_violations(A, red4, blue5, x)
        if v == 0:
            break
        i = int(rng.random(A.shape[0]))
        x[i] = 1 - x[i]
    return x

def _anneal(n, A, red4, blue5, x, t0, deadline, rng):
    best_x = x.copy()
    best_v = _count_violations(A, red4, blue5, x)
    cur_v = best_v
    M = A.shape[0]
    T = 0.5
    # geometric cooling
    steps = 200000
    T_end = 0.001
    cool = (T_end / T) ** (1.0 / steps)
    if cur_v == 0:
        return best_x, best_v
    for _ in range(steps):
        if time.time() >= deadline:
            break
        i = int(rng.random(M))
        x[i] = 1 - x[i]
        nv = _count_violations(A, red4, blue5, x)
        d = nv - cur_v
        if d <= 0 or (T > T_end and rng.random() < np.exp(-d / T)):
            cur_v = nv
            if nv < best_v:
                best_v = nv
                best_x = x.copy()
                if best_v == 0:
                    return best_x, best_v
        else:
            x[i] = 1 - x[i]
        T *= cool
    return best_x, best_v

def _check(n, A, red4, blue5, x):
    return _count_violations(A, red4, blue5, x) == 0

def construct():
    t0 = time.time()
    deadline = t0 + LIMIT
    rng = random.Random(1234567)
    np_rng = np.random.default_rng(1234567)
    best_n = 0
    best_colors = None
    # target the published 34; fall back gracefully to smaller n
    for n in (34, 33, 32, 31, 30, 28, 26, 24, 22, 20, 16, 12, 10, 9):
        if time.time() >= deadline - 2.0:
            break
        triples, A, red4, blue5 = _build(n)
        found = False
        ntries = 0
        while time.time() < deadline - 2.0:
            x = _init(n, A, red4, blue5, np_rng)
            x, v = _anneal(n, A, red4, blue5, x, t0, deadline, np_rng)
            ntries += 1
            if v == 0:
                best_n = n
                best_colors = [int(c) for c in x]
                found = True
                break
        if found:
            break
    if best_n == 0:
        # guaranteed fallback (small n always solvable)
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1
                       for a, b, c in combinations(range(n), 3)]
        best_n = n
    return {'n': best_n, 'colors': best_colors}
