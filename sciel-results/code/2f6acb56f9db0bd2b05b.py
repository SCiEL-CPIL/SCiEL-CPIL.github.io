"""QAP tai256c solver: greedy construction + 2-opt local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _qap_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    pb = b[np.ix_(p, p)]
    return int(np.sum(a * pb))


def _greedy_initial(a, b, n):
    """Greedy placement: place facilities in descending row-sum order, choosing the
    location that minimizes the marginal cost against already-placed facilities."""
    # a_sym[i,j] = a[i,j]+a[j,i] so that a pair {i,k} contributes a[i,k]*d + a[k,i]*d.
    a_sym = a + a.T
    row_sum = a.sum(axis=1)
    order = np.argsort(-row_sum, kind="stable")

    placed = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    # current contribution of each location L to the objective (over placed pairs)
    # loc_cost[L] = sum_{k placed} a_sym[k, *] * b[*, L]
    loc_cost = np.zeros(n, dtype=np.float64)

    for idx in order:
        # best location for idx = argmin over unused L of loc_cost[L] + (self term 0)
        # but loc_cost only accounts for pairs with already-placed facilities,
        # which is exactly the marginal cost of adding idx at L.
        best = int(np.argmin(np.where(used, np.inf, loc_cost)))
        placed[idx] = best
        used[best] = True
        # update loc_cost for future placements: adding idx at best contributes
        # a_sym[idx, m] * b[best, m] to each other location m
        loc_cost += a_sym[idx, :] * b[best, :]

    p = np.empty(n, dtype=np.int64)
    for i in range(n):
        p[i] = placed[i]
    return p


def _local_search_2opt(a, b, p, deadline):
    """2-opt local search: swap locations of two facilities i,k.
    delta = sum_{m != i,k} (a[i,m]+a[m,i])*(b[p[i],p[m]]+b[p[m],p[i]])
                + (a[k,m]+a[m,k])*(b[p[k],p[m]]+b[p[m],p[k]])
              - (a[i,m]+a[m,i])*(b[p[k],p[m]]+b[p[m],p[k]])
              - (a[k,m]+a[m,k])*(b[p[i],p[m]]+b[p[m],p[i]])
    Use a_sym[i,m]=a[i,m]+a[m,i] and b_sym=b+b.T (symmetric distance).
    delta = sum_{m != i,k} a_sym[i,m]*(b_sym[p[i],p[m]]-b_sym[p[k],p[m]])
              + a_sym[k,m]*(b_sym[p[k],p[m]]-b_sym[p[i],p[m]])
    """
    n = len(p)
    a_sym = a + a.T
    b_sym = b + b.T
    p = p.copy()

    # precompute per-facility distance-to-location vectors
    # d_of_i[m] = b_sym[p[i], m]
    dmat = b_sym[p, :]  # dmat[i, m] = b_sym[p[i], m]

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            pi = p[i]
            # precompute d_i = b_sym[pi, :] and d_k candidate per k
            d_i = dmat[i]
            a_i = a_sym[i]
            for k in range(i + 1, n):
                pk = p[k]
                d_k = dmat[k]
                # delta over m != i,k
                # term = a_i[m]*(d_i[m]-d_k[m]) + a_k[m]*(d_k[m]-d_i[m])
                #      = (a_i[m]-a_k[m])*(d_i[m]-d_k[m])
                diff_d = d_i - d_k
                diff_a = a_i - a_sym[k]
                delta = float(np.dot(diff_a, diff_d))
                # subtract m==i and m==k contributions (they are 0 in true sum)
                # at m==i: diff_a[i]*diff_d[i]; at m==k: diff_a[k]*diff_d[k]
                delta -= diff_a[i] * diff_d[i] + diff_a[k] * diff_d[k]
                if delta < -1e-9:
                    # apply swap
                    p[i], p[k] = pk, pi
                    # update dmat rows i,k
                    dmat[i] = b_sym[pk, :]
                    dmat[k] = b_sym[pi, :]
                    improved = True
        if time.time() > deadline:
            break
    return p


def solve(a, b):
    start = time.time()
    n = len(a)
    # safety margin: 120s limit, keep 15s buffer for cost eval + return
    deadline = start + 105.0

    p = _greedy_initial(a, b, n)
    p = _local_search_2opt(a, b, p, deadline)
    return p.tolist()
# EVOLVE-BLOCK-END
