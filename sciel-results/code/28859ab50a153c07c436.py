"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline):
    """First-improvement 2-opt local search with time guard."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p.copy(), cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break
    return p.copy(), cost


def _surrogate_init(a, b):
    """Initialize using a surrogate cost matrix based on row/col sums."""
    n = a.shape[0]
    # Surrogate: cost of assigning facility i to location j is approximated by
    # the interaction of facility i's flow with location j's distance to others
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Build a cost matrix for the assignment problem
    # C[i, j] = a_row_sums[i] * b_row_sums[j]
    C = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _cycle_cover_init(a, b, p_current):
    """
    Formulate QAP as a Minimum Weight Cycle Cover.
    Cost of edge (i, j) = marginal contribution of assigning facility i to location j
    given current neighbors.
    """
    n = a.shape[0]
    # Build cost matrix for cycle cover
    # C[i, j] = sum_k a[i, k] * b[j, p_current[k]] + sum_k a[k, i] * b[p_current[k], j]
    # This approximates the marginal cost of placing facility i at location j
    
    # Efficient computation:
    # C[i, j] = (a[i, :] @ b[:, j]) + (a[:, i] @ b[:, j])
    # But b[:, j] is column j of b
    
    # C = a @ b + a.T @ b  (this is the standard surrogate)
    # But let's use the current assignment to refine:
    # C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    
    # Precompute: for each j, we need b[j, p] and b[p, j]
    # C[i, j] = a[i, :] dot b[j, p] + a[:, i] dot b[p, j]
    
    # Let's compute this efficiently using matrix multiplication
    # a[i, :] is row i of a
    # b[j, p] is row j of b indexed by p
    
    # C = a @ b[:, p] + a.T @ b[:, p]  ... wait, let me think again
    
    # C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    
    # First term: (a @ b.T)[:, j] evaluated at p... no
    # sum_k a[i, k] * b[j, p[k]] = row i of a dotted with b[j, p]
    # b[j, p] is a vector of length n: [b[j, p[0]], b[j, p[1]], ...]
    
    # So first term matrix: C1[i, j] = a[i, :] dot b[j, p]
    # This is a @ (b[:, p]).T  ... no
    # b[j, p] for all j is b[:, p].T? No.
    # b[:, p] has shape (n, n) where column k is b[:, p[k]]
    # b[j, p] is row j of b[:, p].T? Let's see:
    # (b[:, p])[k, j] = b[p[k], j]
    # We want b[j, p[k]] = b.T[p[k], j] = (b.T[:, p])[k, j]
    
    # So b[j, p] as a row vector for all j: matrix M1 where M1[j, k] = b[j, p[k]]
    # M1 = b[:, p].T? No. b[:, p] has shape (n, n) with (b[:, p])[row, col] = b[row, p[col]]
    # So M1 = b[:, p].T gives M1[col, row] = b[row, p[col]], so M1[j, k] = b[k, p[j]]... not quite.
    
    # Let's just do it directly:
    # M1[i, j] = sum_k a[i, k] * b[j, p[k]]
    # = a[i, :] dot b[j, p]
    # b[j, p] is a 1D array. For all j, this is b[:, p].T? 
    # b[:, p] has shape (n, n): (b[:, p])[r, c] = b[r, p[c]]
    # So b[j, p] = (b[:, p])[j, :]
    # M1 = a @ (b[:, p])  -- this gives M1[i, j] = sum_k a[i, k] * (b[:, p])[k, j] = sum_k a[i, k] * b[k, p[j]]
    # That's not what we want. We want sum_k a[i, k] * b[j, p[k]]
    
    # b[j, p[k]] = b.T[p[k], j]
    # So sum_k a[i, k] * b.T[p[k], j] = sum_k a[i, k] * (b.T[:, p])[k, j]
    # = (a @ b.T[:, p])[i, j]
    
    # Second term: M2[i, j] = sum_k a[k, i] * b[p[k], j]
    # b[p[k], j] = (b[p, :])[k, j]
    # sum_k a[k, i] * (b[p, :])[k, j] = (a.T @ b[p, :])[i, j]
    
    # So C = a @ b.T[:, p] + a.T @ b[p, :]
    
    C = a @ b.T[:, p_current] + a.T @ b[p_current, :]
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    return p_new


def _spectral_init(a, b):
    """Spectral initialization: embed facilities and locations, match by nearest neighbor."""
    n = a.shape[0]
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        k = min(10, n)
        idx_a = np.argsort(evals_a)
        fac_emb = evecs_a[:, idx_a[:k]]
        
        idx_b = np.argsort(-evals_b)
        loc_emb = evecs_b[:, idx_b[:k]]
        
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s kill

    best_p = None
    best_cost = np.inf

    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # 1. Cycle Cover Initialization (primary)
    if time.time() < deadline - 20:
        # Start with identity
        p_init = np.arange(n, dtype=np.int64)
        p_cc = _cycle_cover_init(a, b, p_init)
        cost_cc = _cost(a, b, p_cc)
        _update_best(p_cc, cost_cc)
        
        # Iterate cycle cover a few times
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            p_cc_new = _cycle_cover_init(a, b, p_cc)
            cost_cc_new = _cost(a, b, p_cc_new)
            if cost_cc_new < cost_cc:
                p_cc = p_cc_new
                cost_cc = cost_cc_new
                _update_best(p_cc, cost_cc)
            else:
                break
        
        # Local search on cycle cover result
        if time.time() < deadline - 15:
            p_ls, cost_ls = _local_search(a, b, p_cc.copy(), deadline - 15)
            _update_best(p_ls, cost_ls)

    # 2. Surrogate Initialization
    if time.time() < deadline - 15:
        p_sur = _surrogate_init(a, b)
        cost_sur = _cost(a, b, p_sur)
        _update_best(p_sur, cost_sur)
        
        if time.time() < deadline - 10:
            p_ls, cost_ls = _local_search(a, b, p_sur.copy(), deadline - 10)
            _update_best(p_ls, cost_ls)

    # 3. Spectral Initialization
    if time.time() < deadline - 15:
        p_spec = _spectral_init(a, b)
        cost_spec = _cost(a, b, p_spec)
        _update_best(p_spec, cost_spec)
        
        if time.time() < deadline - 10:
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 10)
            _update_best(p_ls, cost_ls)

    # 4. Iterated Local Search from best solution
    if best_p is not None and time.time() < deadline - 5:
        current_p = best_p.copy()
        current_cost = best_cost
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 2:
            # Perturb current solution
            n_swaps = rng.integers(2, 6)
            p_perturbed = current_p.copy()
            for _ in range(n_swaps):
                i, j = rng.integers(0, n, 2)
                if i != j:
                    p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
            # Local search on perturbed solution
            p_ls, cost_ls = _local_search(a, b, p_perturbed, deadline - 2)
            
            # Accept if better
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
