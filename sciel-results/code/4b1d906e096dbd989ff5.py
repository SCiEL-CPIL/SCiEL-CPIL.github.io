"""QAP wil100 solver: rank-matching init + vectorized SA (O(n) swap delta) + 2-opt polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

_DEADLINE = 145.0  # safety margin vs 150s kill


def _total_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _init_rank(a, b):
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def _init_hungarian(a, b):
    n = len(a)
    fa = a.sum(axis=1)
    fb = b.sum(axis=1)
    cost = fa[:, None] * fb[None, :]
    _, cols = linear_sum_assignment(cost)
    p = np.empty(n, dtype=np.int64)
    p[cols] = np.arange(n)
    return p


def _polish_2opt(a, b, p, deadline, max_passes=200):
    n = len(p)
    cost = _total_cost(a, b, p)
    ra = a.sum(axis=1)
    rb = b.sum(axis=1)
    # ra[i] = sum_j a[i,j] ; rb[k] = sum_l b[k,l]
    # swap i<->j (locations p[i],p[j]):
    #  d = (ra[i]+ra[j])*(rb[p[j]]-rb[p[i]]) + (ra[j]-ra[i])*(b[p[i],p[j]]-b[p[j],p[i]])
    #      + (a[i,j]-a[j,i])*(rb[p[i]]+rb[p[j]])
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            rai = ra[i]
            for j in range(i + 1, n):
                pj = p[j]
                raj = ra[j]
                diff = (rai + raj) * (rb[pj] - rb[pi]) \
                    + (raj - rai) * (b[pi, pj] - b[pj, pi]) \
                    + (a[i, j] - a[j, i]) * (rb[pi] + rb[pj])
                if diff < 0:
                    p[i], p[j] = pj, pi
                    cost += int(diff)
                    improved = True
    return p, cost


def solve(a, b):
    start = time.time()
    deadline = start + _DEADLINE
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    ra = a.sum(axis=1)
    rb = b.sum(axis=1)

    # Build initial solution: pick better of rank-matching and Hungarian, then polish.
    p_rank = _init_rank(a, b)
    p_hun = _init_hungarian(a, b)
    c_rank = _total_cost(a, b, p_rank)
    c_hun = _total_cost(a, b, p_hun)
    if c_hun < c_rank:
        p = p_hun.copy()
        cost = c_hun
    else:
        p = p_rank.copy()
        cost = c_rank

    # Initial 2-opt polish to get a good local optimum.
    p, cost = _polish_2opt(a, b, p, deadline, max_passes=60)

    # Simulated annealing with vectorized O(n) swap-delta moves.
    T0 = max(1.0, 0.02 * cost)
    Tend = max(0.01, 1e-6 * cost)
    alpha = (Tend / T0) ** (1.0 / max(1, 200000))

    rng = np.random.default_rng(12345)
    T = T0
    best_p = p.copy()
    best_cost = cost

    while time.time() < deadline:
        # batch of candidate moves
        m = 64
        i = rng.integers(0, n, size=m)
        j = rng.integers(0, n, size=m)
        same = i == j
        if same.any():
            j[same] = (j[same] + 1) % n
        pi = p[i]
        pj = p[j]
        rai = ra[i]
        raj = ra[j]
        diff = (rai + raj) * (rb[pj] - rb[pi]) \
            + (raj - rai) * (b[pi, pj] - b[pj, pi]) \
            + (a[i, j] - a[j, i]) * (rb[pi] + rb[pj])
        diff = diff.astype(np.int64)

        # accept: improving always; worsening with prob exp(-diff/T)
        u = rng.random(m)
        accept = (diff < 0) | (u < np.exp(-np.maximum(diff, 0) / max(T, 1e-12)))
        for idx in np.nonzero(accept)[0]:
            ii = int(i[idx])
            jj = int(j[idx])
            d = int(diff[idx])
            p[ii], p[jj] = p[jj], p[ii]
            cost += d
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        T *= alpha
        if T < Tend:
            T = Tend

    # Final 2-opt polish on the best solution found.
    p_final = best_p.copy()
    p_final, cost_final = _polish_2opt(a, b, p_final, deadline, max_passes=200)
    return p_final.tolist()
# EVOLVE-BLOCK-END
