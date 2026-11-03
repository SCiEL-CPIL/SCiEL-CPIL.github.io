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
    n = len(p)

    mask = np.ones(n, dtype=bool)
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


def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with vectorized 2-opt swap deltas.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            # Vectorized: compute delta for swapping i with all j > i
            pi = p[i]
            # For each j, delta = sum_k (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
            #                   + sum_k (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
            #                   + self terms
            # Precompute row differences
            a_i_row = a[i, :]
            a_col_i = a[:, i]
            b_pi_row = b[pi, :]
            b_pi_col = b[:, pi]

            # For all j at once:
            # diff1[j] = b[p[j], p[all]] - b[pi, p[all]]  -- but this is per-element
            # Let's compute: for j != i, delta(i,j) involves:
            #   sum_k (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
            #   + sum_k (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
            #   + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
            #   + a[i,i]*(b[p[j],p[j]]-b[p[i],p[i]]) + a[j,j]*(b[p[i],p[i]]-b[p[j],p[j]])

            # Vectorized over j:
            # term1: sum_k (a[i,k]-a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
            #   = sum_k a[i,k]*b[p[j],p[k]] - sum_k a[j,k]*b[p[j],p[k]]
            #     - sum_k a[i,k]*b[p[i],p[k]] + sum_k a[j,k]*b[p[i],p[k]]
            #   = (a[i,:] @ b[p[j], p]) ... this gets complex. Let me use the mask approach but vectorized.

            # Simpler vectorized approach:
            # delta(i,j) = sum_k [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi])
            #              - a[j,k]*(b[pj,pk]-b[pi,pk]) - a[k,j]*(b[pk,pj]-b[pk,pi])]
            #            + self terms

            # Precompute:
            # R1[j,k] = b[p[j], p[k]] - b[pi, p[k]]  for all j,k
            # This is: b[np.ix_(p,p)] - b[pi, p[np.newaxis,:]]
            Bp = b[np.ix_(p, p)]  # Bp[j,k] = b[p[j], p[k]]
            diff_B = Bp - b[pi, p[np.newaxis, :]]  # diff_B[j,k] = b[pj,pk] - b[pi,pk]

            # term1[j] = sum_k (a[i,k] - a[j,k]) * diff_B[j,k]
            #          = sum_k a[i,k]*diff_B[j,k] - sum_k a[j,k]*diff_B[j,k]
            #          = a[i,:] @ diff_B[j,:] - (a * diff_B)[j,:].sum(axis=1)
            # Actually: sum_k a[j,k]*diff_B[j,k] = row-wise dot of a[j,:] with diff_B[j,:]
            a_diff_B = a * diff_B  # (n, n): [j,k] = a[j,k]*diff_B[j,k]
            sum_a_diff = a_diff_B.sum(axis=1)  # [j] = sum_k a[j,k]*diff_B[j,k]
            a_i_diff = a[i, :] @ diff_B  # [j] = sum_k a[i,k]*diff_B[j,k]

            term1 = a_i_diff - sum_a_diff  # [j]

            # term2[j] = sum_k (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],pi])
            # Let diff_B2[j,k] = b[p[k],p[j]] - b[p[k],pi]
            # = Bp.T[j,k] - b[:, pi][k] ... wait
            # Bp[k,j] = b[p[k], p[j]], so diff_B2[j,k] = Bp[k,j] - b[p[k], pi]
            # = Bp.T[j,k] - b[:, pi][p]... let me think again
            # diff_B2[j,k] = b[p[k], p[j]] - b[p[k], pi]
            # For fixed j: sum_k (a[k,i]-a[k,j]) * (b[p[k],p[j]] - b[p[k],pi])
            # = sum_k a[k,i]*b[p[k],p[j]] - sum_k a[k,i]*b[p[k],pi]
            #   - sum_k a[k,j]*b[p[k],p[j]] + sum_k a[k,j]*b[p[k],pi]

            # b[p[k], p[j]] for all k,j = Bp.T (since Bp[j,k]=b[p[j],p[k]], so Bp.T[k,j]=b[p[j],p[k]])
            # Wait: Bp[j,k] = b[p[j], p[k]]. So Bp.T[k,j] = b[p[j], p[k]].
            # We need b[p[k], p[j]] = Bp[j,k] (just transpose the indices)
            # So the matrix M[j,k] = b[p[k], p[j]] = Bp.T[k,j]... no.
            # M[j,k] = b[p[k], p[j]]. Bp[k,j] = b[p[k], p[j]]. So M = Bp (just read as M[j,k] = Bp[k,j] = Bp.T[j,k])
            # So M = Bp.T

            # diff_B2[j,k] = Bp.T[j,k] - b[p[k], pi]
            # b[p[k], pi] = b[:, pi][p[k]] = b[p, pi]  (vector of length n)
            b_p_pi = b[p, pi]  # [k] = b[p[k], pi]
            diff_B2 = Bp.T - b_p_pi[np.newaxis, :]  # diff_B2[j,k] = b[p[k],p[j]] - b[p[k],pi]

            # term2[j] = sum_k (a[k,i] - a[k,j]) * diff_B2[j,k]
            #          = sum_k a[k,i]*diff_B2[j,k] - sum_k a[k,j]*diff_B2[j,k]
            a_T_diff_B2 = a.T * diff_B2  # [j,k] = a[k,j] * diff_B2[j,k]... wait
            # a.T[k,j] = a[j,k]. We need a[k,j] which is a.T[j,k].
            # So a_T_diff_B2[j,k] = a.T[j,k] * diff_B2[j,k] = a[k,j] * diff_B2[j,k] ✓
            sum_aT_diff = a_T_diff_B2.sum(axis=1)  # [j] = sum_k a[k,j]*diff_B2[j,k]
            a_i_col_diff = a[:, i] @ diff_B2  # [j] = sum_k a[k,i]*diff_B2[j,k]

            term2 = a_i_col_diff - sum_aT_diff  # [j]

            # Self terms for j:
            # a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
            # + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
            b_pj_pi = b[p, pi]  # [j] = b[p[j], pi]
            b_pi_pj = b[pi, p]  # [j] = b[pi, p[j]]
            b_pj_pj = b[p, p]  # [j] = b[p[j], p[j]]
            b_pi_pi = b[pi, pi]

            self1 = a[i, p] * (b_pj_pi - b_pi_pj)  # [j] = a[i,p[j]]*(b[pj,pi]-b[pi,pj])
            self2 = a[p, i] * (b_pi_pj - b_pj_pi)  # [j] = a[p[j],i]*(b[pi,pj]-b[pj,pi])
            self3 = a[i, i] * (b_pj_pj - b_pi_pi)
            self4 = a[p, p] * (b_pi_pi - b_pj_pj)  # [j] = a[pj,pj]*(b_pi_pi - b_pj_pj)

            delta_all = term1 + term2 + self1 + self2 + self3 + self4
            delta_all[i] = 0  # exclude i itself

            # Find best j > i with negative delta
            mask_j = np.zeros(n, dtype=bool)
            mask_j[i+1:] = True
            valid_deltas = delta_all[mask_j]
            if valid_deltas.size > 0 and valid_deltas.min() < 0:
                best_idx_in_mask = valid_deltas.argmin()
                j = np.where(mask_j)[0][best_idx_in_mask]
                p[i], p[j] = p[j], p[i]
                cost += int(delta_all[j])
                improved = True
                # After swap, Bp changes; we need to recompute for next i
                # But for efficiency, let's just continue and recompute Bp at start of each i loop
                # Actually Bp is computed inside the loop, so it's fresh each i iteration
                # Wait, Bp is computed at the top of the i-loop body, so it's correct

        if not improved:
            break

    return p, cost


def _build_surrogate(a, b, p):
    """
    Build a surrogate cost matrix C[i, l] approximating the marginal cost
    of assigning facility i to location l, holding other assignments fixed.
    """
    n = len(p)
    term1 = a @ b[:, p].T
    term2 = a.T @ b[p, :]
    C = term1 + term2
    return C


def _surrogate_perturb(a, b, p, current_cost, rng, n):
    """
    Surrogate-guided perturbation: identify facilities/locations with high
    contribution to cost and perturb those positions.
    """
    perturbed_p = p.copy()

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    contrib = a_row_sums * b_row_sums[p]

    k = 3 + int(rng.integers(0, 4))
    top_idx = np.argsort(-contrib)[:k]

    for idx in top_idx:
        if rng.random() < 0.5:
            j = rng.integers(0, n)
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]
        else:
            low_idx = np.argsort(contrib)[:k]
            j = low_idx[rng.integers(0, k)]
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]

    return perturbed_p


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

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

    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p


def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement using vectorized delta computation.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    Bp = b[np.ix_(p, p)]

    for i in range(n):
        if time.time() > deadline:
            break
        pi = p[i]
        b_pi_p = b[pi, p]
        diff_B = Bp - b_pi_p[np.newaxis, :]

        a_diff_B = a * diff_B
        sum_a_diff = a_diff_B.sum(axis=1)
        a_i_diff = a[i, :] @ diff_B
        term1 = a_i_diff - sum_a_diff

        b_p_pi = b[p, pi]
        diff_B2 = Bp.T - b_p_pi[np.newaxis, :]
        a_T_diff_B2 = a.T * diff_B2
        sum_aT_diff = a_T_diff_B2.sum(axis=1)
        a_i_col_diff = a[:, i] @ diff_B2
        term2 = a_i_col_diff - sum_aT_diff

        b_pj_pi = b[p, pi]
        b_pi_pj = b[pi, p]
        b_pj_pj = b[p, p]
        b_pi_pi = b[pi, pi]

        self1 = a[i, p] * (b_pj_pi - b_pi_pj)
        self2 = a[p, i] * (b_pi_pj - b_pj_pi)
        self3 = a[i, i] * (b_pj_pj - b_pi_pi)
        self4 = a[p, p] * (b_pi_pi - b_pj_pj)

        delta_all = term1 + term2 + self1 + self2 + self3 + self4
        delta_all[i] = 0

        mask_j = np.zeros(n, dtype=bool)
        mask_j[i+1:] = True
        valid_deltas = delta_all[mask_j]
        if valid_deltas.size > 0:
            min_d = valid_deltas.min()
            if min_d < best_delta:
                best_delta = min_d
                best_i = i
                best_j = np.where(mask_j)[0][valid_deltas.argmin()]

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)

    return p, cost, best_delta


def _soft_min(x, tau=1.0):
    """
    Smooth approximation of min(x) using log-sum-exp.
    """
    x_min = np.min(x)
    return x_min + tau * np.log(np.sum(np.exp((x - x_min) / tau)))


def _doubly_stochastic_init(a, b, n, rng, deadline):
    """
    Initialize a doubly stochastic matrix using spectral alignment,
    then refine with projected gradient descent.
    """
    # Start with spectral-based assignment
    p_spec = _spectral_init(a, b, n, rng)
    
    # Create initial doubly stochastic matrix from the permutation
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p_spec[i]] = 1.0
    
    # Projected gradient descent on the Birkhoff polytope
    # Objective: f(X) = sum_{i,j} A[i,j] * (X^T B X)[i,j]
    # Gradient: 2 * (A * (X^T B X) + A^T * (X^T B X)) @ X  (approximate)
    
    # Actually, let's use a simpler approach:
    # The cost is sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    # = sum_{k,l} b[k,l] * (X^T A X)[k,l]  ... wait, let me re-derive
    
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # In continuous form: sum_{i,j} a[i,j] * sum_{k,l} X[i,k] * b[k,l] * X[j,l]
    # = sum_{k,l} b[k,l] * sum_{i,j} a[i,j] * X[i,k] * X[j,l]
    # = sum_{k,l} b[k,l] * (X^T A X)[k,l]  ... no
    # = sum_{k,l} b[k,l] * (A @ (X.T @ X))[k,l]  ... hmm
    
    # Actually: sum_{i,j} a[i,j] X[i,k] X[j,l] = (X^T A X)[k,l]
    # So cost = sum_{k,l} b[k,l] * (X^T A X)[k,l] = trace(B^T X^T A X)
    
    # Gradient wrt X: 2 * A X B^T + 2 * A^T X B  (for symmetric case simplified)
    # More precisely: d/dX trace(B^T X^T A X) = (A + A^T) X B + (A^T + A) X B^T
    # For symmetric A and B: 2 * A X B
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Barzilai-Borwein step size
    X_old = X.copy()
    grad_old = None
    step_size = 1.0
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Compute gradient
        AXB = a_sym @ X @ b_sym
        grad = 2.0 * AXB
        
        # Compute step size using BB
        if grad_old is not None:
            s = X - X_old
            y = grad - grad_old
            s_norm = np.sum(s * s)
            y_norm = np.sum(y * y)
            sy = np.sum(s * y)
            if sy > 1e-10:
                step_size = s_norm / sy
            elif sy < -1e-10:
                step_size = -s_norm / sy
            else:
                step_size = 1.0
            step_size = max(0.01, min(step_size, 10.0))
        
        # Gradient step
        X_new = X - step_size * grad
        
        # Project onto doubly stochastic polytope (Sinkhorn-Knopp)
        # Normalize rows and columns alternately
        for _ in range(50):
            row_sums = X_new.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_new = X_new / row_sums
            
            col_sums = X_new.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            X_new = X_new / col_sums
        
        X_old = X
        X = X_new
        grad_old = grad
    
    # Extract permutation from X using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p


def _doubly_stochastic_refine(a, b, p, deadline):
    """
    Refine a permutation using doubly stochastic relaxation.
    Start from the permutation matrix, do a few PD steps, extract new permutation.
    """
    n = len(p)
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        X[i, p[i]] = 1.0
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    X_old = X.copy()
    grad_old = None
    step_size = 0.5
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        AXB = a_sym @ X @ b_sym
        grad = 2.0 * AXB
        
        if grad_old is not None:
            s = X - X_old
            y = grad - grad_old
            s_norm = np.sum(s * s)
            y_norm = np.sum(y * y)
            sy = np.sum(s * y)
            if sy > 1e-10:
                step_size = s_norm / sy
            elif sy < -1e-10:
                step_size = -s_norm / sy
            else:
                step_size = 0.5
            step_size = max(0.01, min(step_size, 5.0))
        
        X_new = X - step_size * grad
        
        # Sinkhorn projection
        for _ in range(30):
            row_sums = X_new.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_new = X_new / row_sums
            
            col_sums = X_new.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            X_new = X_new / col_sums
        
        X_old = X
        X = X_new
        grad_old = grad
    
    row_ind, col_ind = linear_sum_assignment(-X)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    return p_new


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # Heuristic 1: Rank-matching
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: Spectral initialization
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Heuristic 4: Doubly stochastic initialization
    if time.time() < deadline - 25.0:
        p4 = _doubly_stochastic_init(a, b, n, rng, deadline - 20.0)
        p4, c4 = _local_search(a, b, p4, deadline)
        if c4 < best_cost:
            best_cost = c4
            best_p = p4.copy()

    # Heuristic 5: Random
    if time.time() < deadline - 10.0:
        p5 = rng.permutation(n).astype(np.int64)
        p5, c5 = _local_search(a, b, p5, deadline)
        if c5 < best_cost:
            best_cost = c5
            best_p = p5.copy()

    # --- Phase 2: Benders-style surrogate-guided master problem ---
    if time.time() < deadline - 15.0:
        surrogate = _build_surrogate(a, b, best_p)
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_surr = np.empty(n, dtype=np.int64)
        p_surr[row_ind] = col_ind
        p_surr, c_surr = _local_search(a, b, p_surr, deadline - 10.0)
        if c_surr < best_cost:
            best_cost = c_surr
            best_p = p_surr.copy()
            surrogate = _build_surrogate(a, b, best_p)

    # --- Phase 3: Iterated Local Search with mixed perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 10.0

    iteration = 0
    total_iterations_expected = 30
    while time.time() < ilse_deadline:
        iteration += 1
        progress = min(1.0, iteration / total_iterations_expected)
        temperature = 1.0 - 0.7 * progress

        r = rng.random()
        if r < 0.2:
            # Doubly stochastic perturbation
            if time.time() < ilse_deadline - 5.0:
                perturbed_p = _doubly_stochastic_refine(a, b, current_p, ilse_deadline - 3.0)
            else:
                perturbed_p = _quantum_perturb(current_p, rng, n, temperature)
        elif r < 0.4:
            perturbed_p = _quantum_perturb(current_p, rng, n, temperature)
        elif r < 0.7:
            perturbed_p = _surrogate_perturb(a, b, current_p, current_cost, rng, n)
        else:
            k = 2 + int(temperature * 4)
            perturbed_p = current_p.copy()
            for _ in range(k):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost < current_cost:
            delta = perturbed_cost - current_cost
            accept_prob = np.exp(-delta / (temperature * 1000.0))
            if rng.random() < accept_prob:
                current_p = perturbed_p
                current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 4: Aggressive final polish with best-improvement ---
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    # --- Phase 5: Final first-improvement polish ---
    if time.time() < deadline - 0.5:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=5)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
