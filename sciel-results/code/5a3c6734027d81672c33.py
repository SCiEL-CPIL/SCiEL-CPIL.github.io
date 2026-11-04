"""QAP tai256c solver: Hungarian-based start + 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    pb = b[np.ix_(p, p)]
    return int(np.sum(a.astype(np.int64) * pb))


def _swap_delta(a, b, p, i, j):
    """Delta in cost if we swap positions i and j in permutation p.
    p maps facility -> location. Swapping p[i] and p[j] changes only
    terms involving facility i or j.
    """
    li, lj = p[i], p[j]
    # terms that change: (i,*) , (*,i), (j,*) , (*,j)
    # new location for i is lj, for j is li
    # contribution before for (i,k): a[i,k]*b[li, p[k]] ; after: a[i,k]*b[lj, p[k]]
    # contribution before for (k,i): a[k,i]*b[p[k], li] ; after: a[k,i]*b[p[k], lj]
    # similarly for j with li.
    # We can compute via vectorized sums over k (all k), but k=i and k=j handled naturally.
    ai = a[i]          # row i: a[i,k]
    aj = a[j]          # row j
    bi = b[li]         # b[li, *]
    bj = b[lj]         # b[lj, *]
    # sum_k a[i,k]*b[li, p[k]] - a[i,k]*b[lj, p[k]] = dot(ai, bi[p] - bj[p])
    # sum_k a[k,i]*b[p[k], li] - a[k,i]*b[p[k], lj] = dot(a[:,i], b[:,li][p] - b[:,lj][p])
    # but b[:,li][p] = b[p, li] (column li indexed by p). Use b[p, li].
    p_loc = p  # array of locations
    delta_i = np.dot(ai, b[p_loc, li] - b[p_loc, lj]) + np.dot(a[:, i], b[li, p_loc] - b[lj, p_loc])
    delta_j = np.dot(aj, b[p_loc, lj] - b[p_loc, li]) + np.dot(a[:, j], b[lj, p_loc] - b[li, p_loc])
    return delta_i + delta_j


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin before 120s kill
    a64 = a.astype(np.int64)
    b64 = b.astype(np.int64)

    # --- Start: Hungarian on flow centrality vs distance centrality ---
    # facility flow centrality: sum of flows involving i
    flow_c = a64.sum(axis=1).astype(np.float64)
    # location distance centrality: sum of distances from i
    dist_c = b64.sum(axis=1).astype(np.float64)
    # We want heavy flow facilities placed at low-distance (central) locations.
    # Minimize sum flow_c[i]*dist_c[p[i]] -> use linear assignment with cost matrix.
    cost_mat = np.outer(flow_c, dist_c)
    # Hungarian minimizes total cost; we want high flow * low dist, so minimize flow*dist.
    _, p = linear_sum_assignment(cost_mat)
    p = p.astype(np.int64)
    # p[k] = location assigned to facility k? linear_sum_assignment returns row->col.
    # row = facility, col = location. So p[facility] = location. Good.

    cur = _qap_cost(a64, b64, p)
    best_p = p.copy()
    best_cost = cur

    # --- 2-opt local search ---
    # Precompute for speed: keep p as list for swapping
    p_list = p.tolist()

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Random-ish order to avoid cycles; use a fixed shuffle each pass
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in idx:
            if time.time() > deadline:
                break
            li = p_list[i]
            for j in idx[i + 1:]:
                if time.time() > deadline:
                    break
                # quick check: skip if same
                if j == i:
                    continue
                d = _swap_delta(a64, b64, np.array(p_list, dtype=np.int64), i, j)
                if d < 0:
                    p_list[i], p_list[j] = p_list[j], p_list[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = np.array(p_list, dtype=np.int64)
            if time.time() > deadline:
                break

    return best_p.tolist()
# EVOLVE-BLOCK-END
