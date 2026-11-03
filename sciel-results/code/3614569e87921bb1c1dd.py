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
    """Best-improvement 2-opt local search with time guard."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
    return p.copy(), cost


def _spectral_init(a, b):
    """Spectral initialization: embed facilities and locations, match by nearest neighbor."""
    n = a.shape[0]
    # Build Laplacian-like matrices for spectral embedding
    # For facilities: use flow matrix a
    # For locations: use distance matrix b
    
    # Symmetrize for eigenvalue computation
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Compute eigenvectors
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        # Use smallest eigenvectors (most "central" embedding)
        # Take top k eigenvectors for embedding
        k = min(10, n)
        # For facilities, use eigenvectors corresponding to smallest eigenvalues
        idx_a = np.argsort(evals_a)
        fac_emb = evecs_a[:, idx_a[:k]]
        
        # For locations, use eigenvectors corresponding to largest eigenvalues (most spread out)
        idx_b = np.argsort(-evals_b)
        loc_emb = evecs_b[:, idx_b[:k]]
        
        # Normalize embeddings
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        # Match by negative inner product (closest embeddings)
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        # Fallback to rank-matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin for 150s limit
    
    # Benders decomposition approach:
    # 1. Start with linearized surrogate: minimize sum_{i,j} a[i,j] * b[p[i],p[j]]
    #    approximated by sum_i (sum_j a[i,j]) * b[p[i], *] + ... 
    #    Actually, the linearized surrogate for facility i at location l is:
    #    sum_j a[i,j] * b[l, p[j]] but p[j] is unknown.
    #    
    #    Better: use the "linear relaxation" where we assign each facility independently
    #    based on weighted average distance. The surrogate cost for facility i at location l
    #    is: sum_j a[i,j] * mean_b[l, :] where mean_b is some aggregate.
    #    
    #    A practical Benders approach:
    #    - Master: solve assignment problem with surrogate cost C[i,l]
    #    - Subproblem: evaluate exact cost, generate cuts
    #    - Cuts: for each facility i and location l, if placing i at l led to high cost,
    #      add a penalty. Use "no-good" cuts or subgradient-based penalties.
    
    # Initialize surrogate cost matrix
    # Linearized surrogate: for facility i at location l, cost ≈ 
    # sum_j a[i,j] * b[l, j] (assuming facility j is at location j as baseline)
    # This is a reasonable linear approximation.
    surrogate = a @ b  # surrogate[i, l] = sum_j a[i,j] * b[j, l]
    
    # Also add the "column" part: sum_j a[j,i] * b[j, l]
    surrogate += a.T @ b  # Now surrogate[i,l] = sum_j (a[i,j] + a[j,i]) * b[j,l]
    
    # Benders iterations
    best_p = None
    best_cost = np.inf
    cuts = np.zeros((n, n), dtype=np.float64)  # Additional penalty from Benders cuts
    
    # Number of Benders iterations to attempt
    max_benders_iters = 20
    
    for benders_iter in range(max_benders_iters):
        if time.time() > deadline - 10:
            break
        
        # Master problem: solve assignment with surrogate + cuts
        master_cost_matrix = surrogate + cuts
        row_ind, col_ind = linear_sum_assignment(master_cost_matrix)
        p_master = np.empty(n, dtype=np.int64)
        p_master[row_ind] = col_ind
        
        # Evaluate exact cost
        exact_cost = _cost(a, b, p_master)
        
        if exact_cost < best_cost:
            best_cost = exact_cost
            best_p = p_master.copy()
        
        # Generate Benders cut:
        # The idea: the exact cost can be decomposed as:
        # sum_i sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i [sum_j a[i,j] * b[p[i], p[j]]]
        # 
        # For a fixed assignment p, the "reduced cost" of facility i at location p[i]
        # is: sum_j a[i,j] * b[p[i], p[j]] - surrogate_linear_part[i, p[i]]
        # 
        # We use this to generate a cut that penalizes assignments that led to high
        # quadratic interaction costs.
        
        # Compute the actual per-facility quadratic contribution
        # For facility i at location l=p[i]:
        # actual_i = sum_j a[i,j] * b[l, p[j]]
        # linear_i = sum_j a[i,j] * b[l, j]  (the surrogate assumes facility j at loc j)
        
        # The "gap" is: actual_i - linear_i = sum_j a[i,j] * (b[l, p[j]] - b[l, j])
        # This gap depends on the full assignment p, so it's not a simple cut.
        
        # Alternative Benders cut: use the exact cost to update a lower bound
        # and adjust the surrogate to guide towards better assignments.
        
        # Subgradient-based cut: for each facility i and location l,
        # estimate the marginal cost of placing i at l given the current assignment.
        # marginal[i, l] = sum_j a[i,j] * b[l, p[j]]  (cost contribution if i were at l)
        
        # Update cuts to penalize locations that are "overused" in terms of
        # quadratic interaction cost.
        
        # Compute marginal costs for the current assignment
        # marginal[i, l] = sum_j a[i,j] * b[l, p[j]]
        # This is: (a @ Bp)[i, :] where Bp[j, k] = b[p[j], p[k]]... 
        # Actually: marginal[i, l] = sum_j a[i,j] * b[l, p[j]]
        # = (a @ b[:, p])[i, :]  -- no, that's not right either.
        # marginal[i, l] = sum_j a[i,j] * b[l, p[j]]
        # Let Bp_row[l, j] = b[l, p[j]], then marginal = a @ Bp_row
        # Bp_row = b[:, p]  (n x n matrix where Bp_row[l, j] = b[l, p[j]])
        
        Bp_row = b[:, p_master]  # Bp_row[l, j] = b[l, p_master[j]]
        marginal = a @ Bp_row  # marginal[i, l] = sum_j a[i,j] * b[l, p_master[j]]
        
        # The surrogate was: surrogate[i, l] = sum_j (a[i,j] + a[j,i]) * b[j, l]
        # The actual "row" contribution is: marginal[i, l] = sum_j a[i,j] * b[l, p[j]]
        
        # For the cut, we want to penalize assignments where the actual cost
        # deviates from the surrogate. Use a simple approach:
        # Add a penalty proportional to the difference between actual and surrogate
        # for the current assignment, distributed across all locations.
        
        # Simpler cut: if the current solution is good, reinforce it; if bad, penalize.
        # Use a "reward" for the current assignment's facility-location pairs.
        
        # For now, use a subgradient step: update cuts based on the dual variables
        # of the assignment problem.
        
        # Get the dual variables from linear_sum_assignment (not directly available)
        # Instead, use a heuristic: penalize locations that are "far" from the
        # optimal assignment in terms of the exact cost landscape.
        
        # Compute the exact cost decomposition:
        # total = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # For each facility i, its "share" of the cost is approximately:
        # share_i = sum_j a[i,j] * b[p[i], p[j]]
        
        Bp_full = b[np.ix_(p_master, p_master)]
        share = a @ Bp_full  # share[i, j] = sum_k a[i,k] * b[p[i], p[j]]... 
        # Actually: (a @ Bp_full)[i, j] = sum_k a[i,k] * Bp_full[k, j] = sum_k a[i,k] * b[p[k], p[j]]
        # That's not quite right. Let me recompute.
        
        # The exact cost is: sum_{i,j} a[i,j] * b[p[i], p[j]]
        # We can write this as: sum_i (sum_j a[i,j] * b[p[i], p[j]])
        # inner_i = sum_j a[i,j] * b[p[i], p[j]]
        # This is: (a @ Bp_cols)[i, i] where Bp_cols[j, k] = b[p[i], p[k]]... 
        # No. Let's just compute it directly.
        
        # inner[i] = sum_j a[i,j] * b[p[i], p[j]]
        # = sum_j a[i,j] * Bp_full[i, j]  where Bp_full[i,j] = b[p[i], p[j]]
        # So inner = np.sum(a * Bp_full, axis=1)
        
        inner = np.sum(a * Bp_full, axis=1)
        
        # The surrogate predicted for facility i at location p[i]:
        # pred[i] = surrogate[i, p[i]] = sum_j (a[i,j] + a[j,i]) * b[j, p[i]]
        
        pred = surrogate[np.arange(n), p_master]
        
        # Gap: inner[i] - pred[i] (could be positive or negative)
        gap = inner - pred
        
        # Use the gap to generate a cut: penalize facility i at locations that
        # are similar to p[i] if the gap is positive (overestimated by surrogate).
        # Simple approach: add a constant penalty to all locations for facility i
        # proportional to the gap.
        
        # Learning rate for cut update
        lr = 0.1
        
        # Update cuts: add lr * gap[i] to cuts[i, :] for all locations
        # This penalizes facility i across all locations if the surrogate overestimated
        # its cost contribution.
        
        cuts += lr * gap[:, np.newaxis]
        
        # Normalize cuts to prevent numerical issues
        cuts -= cuts.min()
        
        # Also try local search from the Benders solution
        if time.time() < deadline - 5:
            p_ls, cost_ls = _local_search(a, b, p_master.copy(), deadline - 5)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls
    
    # If we still have time, try spectral initialization + local search
    if time.time() < deadline - 10:
        p_spec = _spectral_init(a, b)
        cost_spec = _cost(a, b, p_spec)
        if cost_spec < best_cost:
            best_cost = cost_spec
            best_p = p_spec.copy()
        
        if time.time() < deadline - 5:
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 5)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls
    
    # Random restarts if time permits
    if time.time() < deadline - 15:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            p_r = rng.permutation(n)
            cost_r = _cost(a, b, p_r)
            if cost_r < best_cost:
                best_cost = cost_r
                best_p = p_r.copy()
            if time.time() < deadline - 5:
                p_ls, cost_ls = _local_search(a, b, p_r.copy(), deadline - 5)
                if cost_ls < best_cost:
                    best_cost = cost_ls
                    best_p = p_ls
    
    # Final local search from best solution if time permits
    if time.time() < deadline - 2 and best_p is not None:
        p_final, cost_final = _local_search(a, b, best_p.copy(), deadline - 1)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
