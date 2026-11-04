"""Tabu search for the QAPLIB tai256c instance, with a deadline guard."""
import time
import numpy as np


def _cost(a, b, p):
    """Exact integer objective sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int((a @ b[np.ix_(p, p)]).sum())


def _swap_delta(a, b, p, i, j):
    """Delta in objective from swapping locations p[i] and p[j] (O(n))."""
    li = p[i]
    lj = p[j]
    ai = a[i]
    aj = a[j]
    b_li = b[li]
    b_lj = b[lj]
    # Contribution of (i,j) and (j,i) pairs before/after.
    before = ai[lj] * b_li[i] + aj[li] * b_lj[j] + ai[li] * b_li[j] + aj[lj] * b_lj[i]
    after = ai[li] * b_li[i] + aj[lj] * b_lj[j] + ai[lj] * b_li[j] + aj[li] * b_lj[i]
    # All other facilities k: pairs (k,i),(i,k),(k,j),(j,k) change.
    # For (k,i): before a[k,i]*b[p[k],li], after a[k,i]*b[p[k],lj]
    # For (i,k): before a[i,k]*b[li,p[k]], after a[i,k]*b[lj,p[k]]
    # For (k,j): before a[k,j]*b[p[k],lj], after a[k,j]*b[p[k],li]
    # For (j,k): before a[j,k]*b[lj,p[k]], after a[j,k]*b[li,p[k]]
    diff = (ai @ (b_lj - b_li)) + (aj @ (b_li - b_lj)) + ((a[:, i] - a[:, j]) @ (b_lj - b_li))
    # Subtract the double-counted k=i and k=j terms in the last vector expression.
    diff -= (ai - aj) * (b_lj - b_li)[i]
    diff -= (ai - aj) * (b_lj - b_li)[j]
    return after - before + diff


def _tabu_search(a, b, p, tabu_len, iterations, deadline):
    """Improve p via tabu search; returns best permutation and cost."""
    n = len(p)
    p = p.astype(np.int64, copy=True)
    best_p = p.copy()
    best_cost = _cost(a, b, best_p)
    cur_cost = best_cost
    tabu_until = np.zeros(n, dtype=np.int64)
    it = 0
    # Precompute a row-sum-like helper? Keep O(n) per candidate.
    while it < iterations:
        if time.time() >= deadline:
            break
        best_gain = 0
        best_i = -1
        best_j = -1
        best_aspiration = False
        for i in range(n):
            ai = a[i]
            li = p[i]
            b_li = b[li]
            # Vectorize over j to speed up the scan.
            # For each j: compute delta.
            # We'll vectorize the "other facilities" part.
            j_idx = np.arange(n)
            mask = j_idx != i
            js = j_idx[mask]
            lj = p[js]
            # b_lj rows for each j
            b_lj = b[lj]                       # (m, n)
            # term1 = ai @ (b_lj - b_li)  -> (m,)
            term1 = (b_lj - b_li) @ ai
            # term2 = (a[:,i]-a[:,j]) @ (b_lj - b_li)  -> (m,)
            a_diff = a[:, i][None, :] - a[js]           # (m, n)
            term2 = (a_diff * (b_lj - b_li)).sum(axis=1)
            # correction for k=i and k=j
            diff_vec = (b_lj - b_li)[i]
            corr_i = (ai - a[js, i]) * diff_vec
            corr_j = (ai - a[js, i]) * (b_lj - b_li)[js]
            # before/after of the 4 cross terms
            before = (ai[lj] * b_li[i] + a[js, i] * b_lj[i]
                      + ai[li] * b_li[js] + a[js, i] * b_lj[js])
            after = (ai[li] * b_li[i] + a[js, i] * b_lj[i]
                     + ai[lj] * b_li[js] + a[js, i] * b_lj[js])
            delta = after - before + term1 + term2 - corr_i - corr_j
            # Tabu: disallow if tabu_until[i] > it or tabu_until[j] > it, unless aspiration
            tabu_i = tabu_until[i] > it
            tabu_j = tabu_until[js] > it
            allowed = ~(tabu_i | tabu_j)
            # Aspiration: allow if it improves best_cost
            aspiration = (cur_cost + delta) < best_cost
            cand = allowed | aspiration
            if not cand.any():
                continue
            idx = np.where(cand)[0]
            gains = delta[idx]
            k = int(np.argmin(gains))
            g = gains[k]
            if g < best_gain or (g == best_gain and idx[k] < n):
                best_gain = g
                best_j = js[idx[k]]
                best_i = i
                best_aspiration = (cur_cost + best_gain) < best_cost
        if best_i < 0:
            break
        # Apply swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cur_cost += best_gain
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()
        tabu_until[best_i] = it + tabu_len
        tabu_until[best_j] = it + tabu_len
        it += 1
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    # ---- Initial heuristic: greedy matching by row sums ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    c0 = _cost(a, b, p0)

    # ---- Local improvement: 2-opt style hill climbing to a local optimum ----
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 20.0:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p0, i, j)
                if d < 0:
                    p0[i], p0[j] = p0[j], p0[i]
                    c0 += d
                    improved = True
    best_p, best_cost = p0.copy(), c0

    # ---- Tabu search with a few restarts from perturbed starts ----
    tabu_len = max(8, n // 16)
    # Main run
    p_t, c_t = _tabu_search(a, b, best_p, tabu_len, 100000, deadline - 5.0)
    if c_t < best_cost:
        best_p, best_cost = p_t, c_t

    # A couple of restarts with random perturbations if time remains
    rng = np.random.default_rng(12345)
    for restart in range(6):
        if time.time() > deadline - 10.0:
            break
        p_r = best_p.copy()
        # Perturb: random swaps
        for _ in range(n // 4):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                p_r[i], p_r[j] = p_r[j], p_r[i]
        p_t, c_t = _tabu_search(a, b, p_r, tabu_len, 20000, deadline - 5.0)
        if c_t < best_cost:
            best_p, best_cost = p_t, c_t

    return best_p.tolist()
