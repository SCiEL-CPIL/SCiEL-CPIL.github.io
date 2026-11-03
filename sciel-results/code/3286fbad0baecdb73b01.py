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


def _build_surrogate(a, b, p):
    """
    Build a Benders-style surrogate cost matrix C[i, l] approximating
    the marginal cost of assigning facility i to location l.
    """
    n = len(p)
    Bp_col_T = b[:, p].T  # shape (n, n), Bp_col_T[i, l] = b[l, p[i]]
    Bp_row = b[p, :]      # shape (n, n), Bp_row[i, l] = b[p[i], l]
    term1 = a @ Bp_col_T  # term1[i, l] = sum_j a[i, j] * b[l, p[j]]
    term2 = a.T @ Bp_row  # term2[i, l] = sum_j a[j, i] * b[p[j], l]
    C = term1 + term2
    return C


def _spectral_init(a, b):
    """Spectral initialization: embed facilities and locations, match by nearest neighbor."""
    n = a.shape[0]
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        k = min(12, n)
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


def _local_search(a, b, p, deadline, surrogate=None, max_rounds=50):
    """
    First-improvement 2-opt local search with time guard and optional surrogate-guided move ordering.
    Uses vectorized delta computation for speed.
    """
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    rounds = 0
    
    while improved and time.time() < deadline and rounds < max_rounds:
        improved = False
        rounds += 1
        
        for i in range(n):
            if time.time() > deadline:
                return p.copy(), cost
            
            pi = p[i]
            
            # Compute delta for all j > i at once (vectorized)
            # For each j, delta = sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
            #                    + self terms
            
            # Build the vector of deltas for all j
            # diff1[j] = sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]]) for k != i, j
            # This is tricky to fully vectorize per j, so let's do a semi-vectorized approach
            
            # Precompute: for fixed i, the contribution from row i to location l is:
            # row_contrib[l] = sum_{k != i} a[i, k] * b[l, p[k]]
            # Similarly col_contrib[l] = sum_{k != i} a[k, i] * b[p[k], l]
            
            # Actually, let's just iterate j but with numpy for the inner sum
            # This is O(n^2) per round which is 10000 for n=100, acceptable
            
            j_candidates = list(range(i + 1, n))
            
            if surrogate is not None and len(j_candidates) > 40:
                # Sample candidates guided by surrogate
                sample_size = min(40, len(j_candidates))
                # Compute surrogate deltas for all j, pick best sample_size
                surr_deltas = np.array([surrogate[i, p[j]] - surrogate[i, pi] + surrogate[j, pi] - surrogate[j, p[j]] for j in j_candidates])
                top_idx = np.argsort(surr_deltas)[:sample_size]
                j_candidates = [j_candidates[k] for k in top_idx]
            
            for j in j_candidates:
                if time.time() > deadline:
                    return p.copy(), cost
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, restart scan
    
    return p.copy(), cost


def _polish_pass(a, b, p, deadline):
    """
    High-precision polish: full-scan best-improvement 2-opt until no improving swap exists
    or deadline is reached.
    """
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    max_rounds = 20
    
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_pair = (-1, -1)
        
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                delta = _delta_swap(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        
        if best_pair != (-1, -1):
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            cost += best_delta
            improved = True
    
    return p.copy(), cost


def _anneal(a, b, p, deadline, rng, t_start_val=1.0, t_end_val=0.01, alpha=0.999):
    """
    Vectorized simulated annealing with 2-opt moves.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    t = t_start_val
    
    while time.time() < deadline:
        # Pick random pair
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        i, j = int(i), int(j)
        
        delta = _delta_swap(a, b, p, i, j)
        
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        t *= alpha
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin

    best_p = None
    best_cost = np.inf

    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    rng = np.random.default_rng(12345)

    # 1. Spectral Initialization + Local Search
    if time.time() < deadline - 20:
        p_spec = _spectral_init(a, b)
        cost_spec = _cost(a, b, p_spec)
        _update_best(p_spec, cost_spec)
        
        if time.time() < deadline - 15:
            surrogate = _build_surrogate(a, b, p_spec)
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 15, surrogate=surrogate)
            _update_best(p_ls, cost_ls)

    # 2. Benders Master: solve assignment with surrogate
    if best_p is not None and time.time() < deadline - 20:
        surrogate = _build_surrogate(a, b, best_p)
        
        if time.time() < deadline - 15:
            row_ind, col_ind = linear_sum_assignment(surrogate)
            p_master = np.empty(n, dtype=np.int64)
            p_master[row_ind] = col_ind
            cost_master = _cost(a, b, p_master)
            _update_best(p_master, cost_master)
            
            if time.time() < deadline - 10:
                p_ls, cost_ls = _local_search(a, b, p_master.copy(), deadline - 10, surrogate=surrogate)
                _update_best(p_ls, cost_ls)

    # 3. Iterated Local Search with Annealing
    if best_p is not None and time.time() < deadline - 10:
        current_p = best_p.copy()
        current_cost = best_cost
        surrogate = _build_surrogate(a, b, current_p)
        
        iteration = 0
        while time.time() < deadline - 8:
            iteration += 1
            
            # Perturb using surrogate-guided reassignment with noise
            if time.time() < deadline - 6:
                noise_scale = 0.15 * (np.max(surrogate) - np.min(surrogate) + 1e-9)
                surrogate_perturbed = surrogate + noise_scale * rng.standard_normal((n, n))
                row_ind, col_ind = linear_sum_assignment(surrogate_perturbed)
                p_perturbed = np.empty(n, dtype=np.int64)
                p_perturbed[row_ind] = col_ind
            else:
                p_perturbed = current_p.copy()
                n_swaps = rng.integers(4, 10)
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, 2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
            # Local search on perturbed
            if time.time() < deadline - 4:
                p_ls, cost_ls = _local_search(a, b, p_perturbed, deadline - 4, surrogate=surrogate, max_rounds=20)
            else:
                p_ls, cost_ls = p_perturbed, _cost(a, b, p_perturbed)
            
            # Accept if better
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)
                surrogate = _build_surrogate(a, b, current_p)
            else:
                # Occasionally update surrogate for diversification
                if rng.random() < 0.4:
                    surrogate = 0.5 * surrogate + 0.5 * _build_surrogate(a, b, p_ls)

    # 4. Final annealing pass on best solution
    if best_p is not None and time.time() < deadline - 3:
        p_ann, cost_ann = _anneal(a, b, best_p.copy(), deadline - 1, rng)
        _update_best(p_ann, cost_ann)

    # 5. Final high-precision polish
    if best_p is not None and time.time() < deadline - 0.5:
        p_polish, cost_polish = _polish_pass(a, b, best_p.copy(), deadline - 0.2)
        _update_best(p_polish, cost_polish)

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
