# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement


def kissing_number11() -> np.ndarray:
    """
    Builds a large set of integer 11-dimensional points with integral coordinates such
    that the minimum squared pairwise distance is at least the maximum squared norm.

    Strategy:
      * Work with vectors of a fixed squared norm R so max_norm2 == R and the validity
        condition reduces to "every pair has squared distance >= R" (inner product <= 0).
      * Build the compatibility graph, then greedily pack a clique with random restarts.
      * Try the small fast pool (R=2) first, then the larger pool (R=4) only if there is
        spare wall-clock time.
      * A hard deadline guard returns the best valid result found with a safety margin.

    Returns:
        points: np.ndarray of shape (num_points, 11) with int64 entries.
    """
    d = 11
    start = time.time()
    deadline = start + 600.0
    safety = 25.0  # seconds to keep clear of the 630s kill

    # ------------------------------------------------------------------
    # Candidate pools (integer vectors of fixed squared norm R)
    # ------------------------------------------------------------------
    def pool_R2():
        # 220 vectors: 22 unit vectors + 2*11*10 mixed pairs (+-1,+-1)
        out = []
        for i in range(d):
            for s in (1, -1):
                v = [0] * d
                v[i] = s
                out.append(v)
        for i in range(d):
            for j in range(i + 1, d):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * d
                        v[i] = si
                        v[j] = sj
                        out.append(v)
        return out

    def pool_R4():
        # 5302 vectors: 44 (4,0,..) + 440 (2,2,0,..) + 1760 (2,1,1,..) + 2856 (1,1,1,1,..)
        out = []
        for i in range(d):
            for s in (2, -2):
                v = [0] * d
                v[i] = s
                out.append(v)
        for i in range(d):
            for j in range(i + 1, d):
                for si in (2, -2):
                    for sj in (2, -2):
                        v = [0] * d
                        v[i] = si
                        v[j] = sj
                        out.append(v)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    if i == j or j == k or i == k:
                        continue
                    for si in (2, -2):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                v = [0] * d
                                v[i] = si
                                v[j] = sj
                                v[k] = sk
                                out.append(v)
        for a, b, c, e in combinations_with_replacement(range(d), 4):
            for s in (1, -1):
                v = [0] * d
                v[a] = s
                v[b] = s
                v[c] = s
                v[e] = s
                out.append(v)
        return out

    # ------------------------------------------------------------------
    # Greedy clique packing with random restarts
    # ------------------------------------------------------------------
    def greedy_clique(pool, R, time_budget):
        n = len(pool)
        if n < 2:
            return None
        # compat[i] = bitset of j > i with (pool[i]-pool[j])^2 >= R
        compat = [None] * n
        for i in range(n):
            pi = pool[i]
            bit = 0
            for j in range(i + 1, n):
                pj = pool[j]
                diff2 = 0
                for k in range(d):
                    t = pi[k] - pj[k]
                    diff2 += t * t
                if diff2 >= R:
                    bit |= (1 << j)
            compat[i] = bit
        # degrees (for a light degree-based ordering)
        deg = [0] * n
        for i in range(n):
            deg[i] = compat[i].bit_count()

        best = []
        best_len = 0
        t0 = time.time()
        rng = random.Random(12345)
        restarts = 0
        while time.time() - t0 < time_budget:
            restarts += 1
            order = list(range(n))
            rng.shuffle(order)
            # prefer higher-degree nodes early (stable-ish heuristic)
            order.sort(key=lambda x: (-deg[x], rng.random()))

            clique = []
            avail = (1 << n) - 1
            for idx in order:
                if (avail >> idx) & 1:
                    # add idx, intersect with its neighbours
                    clique.append(idx)
                    avail &= compat[idx]
                    if avail == 0:
                        break
            if len(clique) > best_len:
                best_len = len(clique)
                best = list(clique)
                if best_len >= n:  # perfect
                    break
        if best_len < 2:
            return None
        return [tuple(pool[i]) for i in best]

    # ------------------------------------------------------------------
    # Driver: try small pool, then large pool if time remains
    # ------------------------------------------------------------------
    best_points = None
    best_len = 0

    # Pool 1: R=2 (fast, 220 vectors)
    pool = pool_R2()
    R = 2
    budget = min(120.0, max(5.0, deadline - time.time() - safety))
    res = greedy_clique(pool, R, budget)
    if res is not None and len(res) > best_len:
        best_len = len(res)
        best_points = res

    # Pool 2: R=4 (larger, 5302 vectors) only if enough time remains
    if time.time() < deadline - safety - 60.0:
        pool = pool_R4()
        R = 4
        budget = max(5.0, deadline - time.time() - safety)
        res = greedy_clique(pool, R, budget)
        if res is not None and len(res) > best_len:
            best_len = len(res)
            best_points = res

    # Fallback: guaranteed-valid 2-point set
    if best_points is None or best_len < 2:
        best_points = [(1,) * d, (-1,) * d]

    arr = np.array(best_points, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr


# EVOLVE-BLOCK-END
