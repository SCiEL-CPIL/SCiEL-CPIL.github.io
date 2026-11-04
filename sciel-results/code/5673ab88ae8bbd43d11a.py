"""
Structural change: replace ILS with a vectorized Simulated Annealing core that
evaluates every 2-swap in a single 100x100 NumPy operation per step, plus a
fast final first-improvement polish.  The initialization uses a Hungarian
matching on a low-rank surrogate built from row-centrality of A and B.
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])

    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_t1 = np.dot(diff_a_row, diff_b_col)
    sum_t1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_t1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])

    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_t2 = np.dot(diff_a_col, diff_b_row)
    sum_t2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_t2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])

    return int(delta)


def _vectorized_deltas(a, b, p):
    """Return a (n, n) matrix of deltas for swapping p[i] and p[j]."""
    n = len(p)
    # Precompute for each pair (i, j):
    # delta = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #       + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #       + sum_k (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
    #       + sum_k (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])
    # We vectorize by computing intermediate matrices.
    bp = b[p]  # bp[i] = b[p[i], :]  (row of B at location p[i])
    # For term 3: sum_k (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
    # = sum_k (a[i,k]-a[j,k])*(bp[pj, pk] - bp[pi, pk])
    # Note bp[pj, pk] = b[pj, p[pk]], but pk is index in p, so p[pk] is the location.
    # Actually bp is b[p, :]. So bp[pj, pk] is b[p[pj], p[pk]]? No.
    # Let's be careful. p is array of length n. p[i] is location of facility i.
    # bp = b[p] means bp[i, k] = b[p[i], k].
    # We need b[pj, p[k]] where pj is a location (p[j]) and p[k] is a location.
    # So b[pj, p[k]] = b[p[j], p[k]].
    # bp[pj, pk] = b[p[pj], pk]. That's not right.
    # Let's define Bp = b[np.ix_(p, p)] which is (n,n) with Bp[i,j] = b[p[i], p[j]].
    # Then b[pj, p[k]] = Bp[j, k]. And b[pi, p[k]] = Bp[i, k].
    # So term 3 = sum_k (a[i,k]-a[j,k]) * (Bp[j,k] - Bp[i,k])
    # = (a[i,:]-a[j,:]) dot (Bp[j,:]-Bp[i,:])
    # This can be computed as a matrix of size (n,n) where entry (i,j) is the dot product.
    # Let D_a[i,k] = a[i,k] and D_bp[i,k] = Bp[i,k].
    # We want M1[i,j] = sum_k (a[i,k]-a[j,k])*(Bp[j,k]-Bp[i,k])
    # = sum_k a[i,k]*Bp[j,k] - sum_k a[i,k]*Bp[i,k] - sum_k a[j,k]*Bp[j,k] + sum_k a[j,k]*Bp[i,k]
    # = (a @ Bp.T)[i,j] - (a * Bp).sum(axis=1)[i] - (a @ Bp.T)[j,i] + (a * Bp).sum(axis=1)[j]
    # Wait, (a @ Bp.T)[i,j] = sum_k a[i,k]*Bp.T[k,j] = sum_k a[i,k]*Bp[j,k]. Yes.
    # And (a * Bp).sum(axis=1)[i] = sum_k a[i,k]*Bp[i,k]. Yes.
    Bp = b[np.ix_(p, p)]
    aBpT = a @ Bp.T  # (n, n)
    aBp_diag = np.sum(a * Bp, axis=1)  # (n,)
    M1 = aBpT - aBp_diag[:, None] - aBpT.T + aBp_diag[None, :]
    # M1[i,j] = sum_k (a[i,k]-a[j,k])*(Bp[j,k]-Bp[i,k])

    # Term 4: sum_k (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
    # b[p[k],pj] = b[p[k], p[j]] = Bp[k, j]
    # b[p[k],pi] = b[p[k], p[i]] = Bp[k, i]
    # So term 4 = sum_k (a[k,i]-a[k,j])*(Bp[k,j]-Bp[k,i])
    # = sum_k a[k,i]*Bp[k,j] - sum_k a[k,i]*Bp[k,i] - sum_k a[k,j]*Bp[k,j] + sum_k a[k,j]*Bp[k,i]
    # = (a.T @ Bp)[i,j] - (a * Bp).sum(axis=0)[i] - (a * Bp).sum(axis=0)[j] + (a.T @ Bp)[j,i]
    # (a.T @ Bp)[i,j] = sum_k a.T[i,k]*Bp[k,j] = sum_k a[k,i]*Bp[k,j]. Yes.
    aT_Bp = a.T @ Bp  # (n, n)
    aBp_col = np.sum(a * Bp, axis=0)  # (n,)
    M2 = aT_Bp - aBp_col[:, None] - aBp_col[None, :] + aT_Bp.T
    # M2[i,j] = sum_k (a[k,i]-a[k,j])*(Bp[k,j]-Bp[k,i])

    # Term 1: (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj]) = (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])
    M3 = (a - a.T) * (Bp.T - Bp)  # (n,n) where M3[i,j] = (a[i,j]-a[j,i])*(Bp[j,i]-Bp[i,j])

    # Term 2: a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    # = (a[i,i]-a[j,j])*(Bp[j,j]-Bp[i,i])
    a_diag = np.diag(a)
    Bp_diag = np.diag(Bp)
    M4 = (a_diag[:, None] - a_diag[None, :]) * (Bp_diag[None, :] - Bp_diag[:, None])

    deltas = M1 + M2 + M3 + M4
    # Set diagonal to 0 (swapping i with i is not allowed)
    np.fill_diagonal(deltas, 0)
    return deltas


def _hungarian_initial(a, b):
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    ars_norm = a_row_sums / ars_max if ars_max > 0 else a_row_sums
    brs_norm = b_row_sums / brs_max if brs_max > 0 else b_row_sums
    C = -np.outer(ars_norm, brs_norm)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _sa_polish(a, b, p, time_limit):
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    start = time.time()
    T_start = 100.0
    T_end = 0.01
    deadline = time_limit

    while time.time() - start < deadline:
        deltas = _vectorized_deltas(a, b, p)
        i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
        if i == j:
            break
        min_delta = deltas[i, j]
        elapsed = time.time() - start
        T = T_start * (T_end / T_start) ** (elapsed / deadline)
        if min_delta < 0:
            accept = True
        else:
            if T > 0:
                prob = np.exp(-min_delta / T)
            else:
                prob = 0.0
            accept = np.random.rand() < prob
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += min_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # If best move is not accepted, try a random move
            ri = np.random.randint(n)
            rj = np.random.randint(n)
            if ri != rj:
                rd = deltas[ri, rj]
                if rd < 0:
                    accept_r = True
                else:
                    if T > 0:
                        prob_r = np.exp(-rd / T)
                    else:
                        prob_r = 0.0
                    accept_r = np.random.rand() < prob_r
                if accept_r:
                    p[ri], p[rj] = p[rj], p[ri]
                    current_cost += rd
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    return best_p, best_cost


def _first_improvement_ls(a, b, p, time_limit):
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start = time.time()
    while time.time() - start < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start > time_limit:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    deadline = 115.0
    start = time.time()
    np.random.seed(42)
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost

    # Phase 1: SA polish
    sa_limit = min(60.0, deadline - (time.time() - start) - 20.0)
    if sa_limit > 1:
        p_sa, cost_sa = _sa_polish(a, b, p, sa_limit)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
        p = p_sa.copy()

    # Phase 2: First improvement LS
    ls_limit = deadline - (time.time() - start) - 5.0
    if ls_limit > 1:
        p_ls, cost_ls = _first_improvement_ls(a, b, p, ls_limit)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    return best_p.tolist()
