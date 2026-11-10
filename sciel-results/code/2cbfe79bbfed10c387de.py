import numpy as np
import random
import time

N = 256

# Precompute all 4-term arithmetic progressions as 0-based index tuples
_AP_LIST = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        i1 = a - 1
        i2 = a + d - 1
        i3 = a + 2 * d - 1
        i4 = a + 3 * d - 1
        _AP_LIST.append((i1, i2, i3, i4))

# Stack as int32 array for fast numpy indexing
_AP_ARR = np.array(_AP_LIST, dtype=np.int32)  # shape (M, 4)
_M = _AP_ARR.shape[0]


def _count_mono(c):
    """c: numpy array of 0/1 ints, length 256. Return count of monochromatic 4-APs."""
    c1 = c[_AP_ARR[:, 0]]
    c2 = c[_AP_ARR[:, 1]]
    c3 = c[_AP_ARR[:, 2]]
    c4 = c[_AP_ARR[:, 3]]
    # monochromatic iff all equal
    return int(np.sum((c1 == c2) & (c2 == c3) & (c3 == c4)))


def _greedy_improve(c, rng, deadline):
    """Local search: repeatedly flip the position with best single-flip gain."""
    cur = _count_mono(c)
    best = cur
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # evaluate each single flip's effect via full recompute is too slow (256 * M)
        # Instead, use incremental: precompute per-position contribution is complex;
        # use random candidate flips with acceptance (fast path).
        # Try a batch of random flips, keep improving ones.
        for _ in range(2000):
            if time.time() >= deadline:
                break
            p = rng.randrange(N)
            old = c[p]
            c[p] = 1 - old
            new = _count_mono(c)
            if new < cur:
                cur = new
                if new < best:
                    best = new
                improved = True
            else:
                c[p] = old
    return best


def _sim_anneal(c, rng, deadline):
    """Simulated annealing over single flips."""
    cur = _count_mono(c)
    best = cur
    best_c = c.copy()
    T = 2.0
    T_min = 0.05
    cooling = 0.999
    while time.time() < deadline and T > T_min:
        T *= cooling
        p = rng.randrange(N)
        old = c[p]
        c[p] = 1 - old
        new = _count_mono(c)
        delta = new - cur
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            cur = new
            if new < best:
                best = new
                best_c = c.copy()
        else:
            c[p] = old
    return best, best_c


def _init_coloring(rng, mode):
    if mode == 0:
        return np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
    elif mode == 1:
        # Thue-Morse-like: parity of popcount
        return np.array([bin(i).count("1") % 2 for i in range(N)], dtype=np.int8)
    elif mode == 2:
        # blocks of 4: 0011
        return np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
    elif mode == 3:
        # random
        return np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)
    elif mode == 4:
        # period 5: 00011
        return np.array([(i % 5) >= 3 for i in range(N)].astype(np.int8))
    else:
        return np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8)


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    rng = random.Random(12345)
    np_rng = np.random.default_rng(777)

    best_score = None
    best_c = None

    # Try a variety of structured starts, improve each
    for mode in range(6):
        if time.time() > deadline:
            break
        c = _init_coloring(rng, mode)
        s = _count_mono(c)
        if best_score is None or s < best_score:
            best_score = s
            best_c = c.copy()

    # Greedy improvement from the best structured start
    if best_c is not None and time.time() < deadline:
        c = best_c.copy()
        s = _greedy_improve(c, rng, deadline)
        if s < best_score:
            best_score = s
            best_c = c.copy()

    # Simulated annealing from multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        c = np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)
        s, best_c_sa = _sim_anneal(c, rng, deadline)
        if s < best_score:
            best_score = s
            best_c = best_c_sa.copy()

    # Final greedy polish from the best found
    if best_c is not None and time.time() < deadline - 5:
        c = best_c.copy()
        s = _greedy_improve(c, rng, deadline)
        if s < best_score:
            best_score = s
            best_c = c.copy()

    if best_c is None:
        return [(i // 2) % 2 for i in range(N)]

    return [int(x) for x in best_c]
