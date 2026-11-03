"""QAP wil100: Hungarian-surrogate init + vectorized 2-opt simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]  (p maps facility -> location)
    return int(np.sum(a * b[np.ix_(p, p)]))


def _surrogate_init(a, b):
    """
    Linear surrogate: assign facility i to location j minimizing
    sum_i (row-flow-centrality_i) * (col-distance-centrality_j)  style, but we
    build a full assignment cost matrix C[i,j] and solve the exact assignment.

    A good linear surrogate for QAP is:
        C[i, j] = (sum_k a[i,k]) * (sum_l b[j,l])  ... but that's separable.
    Better: use the "rank" / "centroid" surrogate where the linear part of the
    QAP objective, when expanded, gives per-(i,j) weights based on the
    interaction of a's row sums with b's column sums is too weak.

    Instead use the standard linearization: the QAP objective
        sum_{i,j} a[i,j] b[p_i, p_j]
    can be lower/upper bounded by a linear assignment if we fix one factor.
    A common strong surrogate: C[i,j] = sum_k a[i,k] * b[j, pi(k)] for some pi.
    We bootstrap: use rank-matching pi0, build C, solve Hungarian -> better pi.
    Iterate a few times (this is the "linear surrogate" fixed-point).
    """
    n = len(a)
    # initial rank-matching permutation
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # iterate surrogate refinement a few times
    for _ in range(3):
        # C[i,j] = sum_k a[i,k] * b[j, p[k]]  (linear cost of putting i at j
        # given the rest of the assignment is p)
        # C = a @ b[:, p]
        C = a @ b[:, p]
        # we want to MINIMIZE sum_i C[i, p[i]]; Hungarian minimizes sum C[i, pi[i]]
        ii, jj = linear_sum_assignment(C)
        p = jj
    return p


def _swap_delta(a, b, p, i, j):
    """
    Delta in cost when swapping the locations of facilities i and j.
    p maps facility -> location. Let li = p[i], lj = p[j].
    After swap, facility i goes to lj, facility j goes to li.

    cost = sum_{x,y} a[x,y] b[p[x], p[y]]
    Only terms involving p[i] or p[j] change (i.e., where x in {i,j} or y in {i,j}).

    Let A = a[:, i] + a[:, j]  (column sums into i or j) -> but careful with double count.
    We compute the change exactly:
    old contribution of rows i,j and cols i,j:
      T_old = a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
             + (sum_{x not in {i,j}} a[x,i]*b[p[x],li]) + (sum_{x not in {i,j}} a[x,j]*b[p[x],lj])
             + (sum_{y not in {i,j}} a[i,y]*b[li,p[y]]) + (sum_{y not in {i,j}} a[j,y]*b[lj,p[y]])
    new (i->lj, j->li):
      T_new = a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
             + (sum_{x not in {i,j}} a[x,i]*b[p[x],lj]) + (sum_{x not in {i,j}} a[x,j]*b[p[x],li])
             + (sum_{y not in {i,j}} a[i,y]*b[lj,p[y]]) + (sum_{y not in {i,j}} a[j,y]*b[li,p[y]])

    delta = T_new - T_old.
    We compute with vector ops over the whole arrays for speed (n=100, cheap).
    """
    li = p[i]
    lj = p[j]
    # vectors of locations of all facilities
    loc = p  # loc[x] = p[x]
    # For a fixed facility x, its contribution to rows of i and j:
    # col_into_i[x] = a[x,i] * b[loc[x], li]  (old) ; a[x,i]*b[loc[x],lj] (new)
    # col_into_j[x] = a[x,j] * b[loc[x], lj]  (old) ; a[x,j]*b[loc[x],li] (new)
    # row_from_i[y] = a[i,y] * b[li, loc[y]]  (old) ; a[i,y]*b[lj, loc[y]] (new)
    # row_from_j[y] = a[j,y] * b[lj, loc[y]]  (old) ; a[j,y]*b[li, loc[y]] (new)

    ai = a[:, i]
    aj = a[:, j]
    bi_row_i = b[li, :]
    bj_row_i = b[lj, :]

    # old: sum_x ai[x]*b[loc[x],li] + sum_x aj[x]*b[loc[x],lj] + sum_y a[i,y]*b[li,loc[y]] + sum_y a[j,y]*b[lj,loc[y]]
    old = (ai * b[loc, li]).sum() + (aj * b[loc, lj]).sum() \
        + (a[i, :] * bi_row_i[loc]).sum() + (a[j, :] * bj_row_i[loc]).sum()
    # new: sum_x ai[x]*b[loc[x],lj] + sum_x aj[x]*b[loc[x],li] + sum_y a[i,y]*b[lj,loc[y]] + sum_y a[j,y]*b[li,loc[y]]
    new = (ai * b[loc, lj]).sum() + (aj * b[loc, li]).sum() \
        + (a[i, :] * bj_row_i[loc]).sum() + (a[j, :] * bi_row_i[loc]).sum()
    return new - old


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under 120s

    p = _surrogate_init(a, b)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)

    # precompute nothing heavy; n=100 so per-swap is O(n) with numpy overhead.
    # Vectorize over pairs to evaluate many candidate swaps at once.

    rng = np.random.default_rng(0)

    # Simulated annealing with 2-opt (single swap of two facilities' locations)
    T = 1.0
    T_min = 0.01
    # adaptive cooling
    step = 0
    while time.time() < deadline:
        # batch of candidate swaps
        batch = 500
        idx_i = rng.integers(0, n, size=batch)
        idx_j = rng.integers(0, n, size=batch)
        # avoid i==j
        same = idx_i == idx_j
        if same.any():
            idx_j[same] = (idx_j[same] + 1) % n

        # evaluate deltas for the batch
        deltas = np.empty(batch, dtype=np.int64)
        for k in range(batch):
            deltas[k] = _swap_delta(a, b, p, idx_i[k], idx_j[k])

        # accept according to SA
        # current cost
        cur_cost = _qap_cost(a, b, p)
        for k in range(batch):
            if time.time() > deadline:
                break
            d = deltas[k]
            i, j = idx_i[k], idx_j[k]
            accept = False
            if d <= 0:
                accept = True
            else:
                # probability exp(-d/T); d is integer, T float
                if d < 1e6:  # avoid overflow
                    if rng.random() < np.exp(-d / max(T, 1e-9)):
                        accept = True
            if accept:
                p[i], p[j] = p[j], p[i]
                cur_cost += d
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
        # cool
        T = max(T_min, T * 0.999)
        step += 1

    # final: make sure best_p is returned
    return best_p.tolist()
# EVOLVE-BLOCK-END
