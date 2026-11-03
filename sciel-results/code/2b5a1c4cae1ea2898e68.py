"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _swap_delta(i, j, p, a, b, bcol):
    """Delta in cost when swapping the two facilities currently at positions i and j."""
    pi = p[i]
    pj = p[j]
    # term 1: flow between i and j, distance changes sign
    t1 = (a[i, j] + a[j, i]) * (b[pi, pj] - b[pj, pi])
    # term 2: combined flow row against all other locations
    flow = a[i] + a[j]  # a[i,k]+a[j,k] for all k
    # distance row: b[pi, :] - b[pj, :]
    dist = b[pi] - b[pj]
    t2 = flow @ dist
    # subtract the i<->j contribution already counted in t2 (k=i and k=j)
    # For k=i: flow[i]=a[i,i]+a[j,i], dist[i]=b[pi,pi]-b[pj,pi]
    # For k=j: flow[j]=a[i,j]+a[j,j], dist[j]=b[pi,pj]-b[pj,pj]
    t2 -= (a[i, i] + a[j, i]) * (b[pi, pi] - b[pj, pi])
    t2 -= (a[i, j] + a[j, j]) * (b[pi, pj] - b[pj, pj])
    return t1 + t2


def _local_search(p, a, b, max_rounds, deadline):
    """Steepest-descent 2-opt. Returns improved permutation and its cost."""
    n = len(p)
    p = p.copy()
    # current cost
    c = 0
    for i in range(n):
        c += a[i] @ b[p[i], p]
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() >= deadline:
            break
        improved = False
        rounds += 1
        best_d = 0
        best_i = -1
        best_j = -1
        # sample-based steepest descent: scan all pairs but cap work via early time check
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta(i, j, p, a, b, b)
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_j = j
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            c += best_d
            improved = True
    return p, c


def _perturb(p, k, rng):
    """Apply k random swaps (2-opt perturbation)."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    start = time.time()
    deadline = start + 110.0  # 10s safety margin before 120s kill
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    rng = np.random.default_rng(12345)

    # --- Build diverse initial solutions ---
    inits = []

    # 1. Rank-matching: heavy flow rows -> central (low distance sum) locations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    inits.append(p1)

    # 2. Reverse-rank: heavy flow rows -> peripheral (high distance sum) locations
    periph = np.argsort(-b.sum(axis=1), kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = periph
    inits.append(p2)

    # 3. Random
    p3 = rng.permutation(n).astype(np.int64)
    inits.append(p3)

    # 4. Greedy row-insertion: place facilities one by one at the location minimizing
    #    incremental cost given already-placed facilities.
    #    cost contribution of placing facility i at location L given placed set S:
    #    sum_{j in S} a[i,j]*b[L,p[j]] + sum_{j in S} a[j,i]*b[p[j],L]
    #    We do this O(n^2) by iterating.
    placed = np.full(n, -1, dtype=np.int64)
    p4 = np.empty(n, dtype=np.int64)
    for fac in range(n):
        best_loc = -1
        best_val = np.inf
        for loc in range(n):
            if placed[loc] >= 0:
                continue
            # incremental cost
            val = 0
            for j in range(n):
                if placed[j] >= 0:
                    val += a[fac, j] * b[loc, placed[j]] + a[j, fac] * b[placed[j], loc]
            if val < best_val:
                best_val = val
                best_loc = loc
        if best_loc < 0:
            best_loc = np.where(placed < 0)[0][0]
        placed[best_loc] = fac
        p4[fac] = best_loc
    inits.append(p4)

    # --- Evaluate each init, keep best after a short local search ---
    best_p = None
    best_c = np.inf
    for init in inits:
        if time.time() >= deadline - 30.0:
            break
        p, c = _local_search(init, a, b, max_rounds=20, deadline=deadline - 20.0)
        if c < best_c:
            best_c = c
            best_p = p.copy()

    # --- ILS: perturb best, re-optimize, keep if better ---
    ils_round = 0
    while time.time() < deadline - 10.0:
        ils_round += 1
        # increase perturbation strength slowly
        k = min(3 + ils_round // 5, 15)
        cand = _perturb(best_p, k, rng)
        p, c = _local_search(cand, a, b, max_rounds=30, deadline=deadline - 5.0)
        if c < best_c:
            best_c = c
            best_p = p.copy()
            # reset perturbation strength on improvement
            ils_round = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
