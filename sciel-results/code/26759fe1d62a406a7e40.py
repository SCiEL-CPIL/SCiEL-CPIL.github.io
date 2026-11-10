# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _fast_ratio(P):
    """
    Fast computation of D/M using vectorized bincount-style approach.
    P[y, x] is the joint probability for x, y in 0..9.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))
    flat_P = P.flatten()

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=flat_P, minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=flat_P, minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=flat_P, minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=flat_P, minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M

def _objective_and_grad(x_flat):
    """
    Objective function to minimize: -ratio(P)
    Returns (value, gradient)
    """
    P = x_flat.reshape(10, 10)
    
    # Ensure P is a valid probability distribution
    P = np.maximum(P, 0)
    s = P.sum()
    if s < 1e-12:
        return 0.0, np.zeros_like(x_flat)
    P = P / s
    
    # Compute entropies
    y_idx, x_idx = np.indices((10, 10))
    
    # Helper to compute entropy and its gradient w.r.t P
    def entropy_and_grad(mapping, num_bins):
        """
        mapping: 10x10 array of bin indices
        num_bins: number of bins
        Returns: entropy, gradient (10x10)
        """
        # counts[b] = sum(P[mapping == b])
        counts = np.zeros(num_bins)
        for b in range(num_bins):
            mask = (mapping == b)
            counts[b] = P[mask].sum()
        
        total = counts.sum()
        if total < 1e-12:
            return 0.0, np.zeros_like(P)
        
        probs = counts / total
        # H = - sum p log2 p
        # dH/dP_ij = - (1/ln 2) * (log2(p_k) + 1) / N * delta_{mapping(ij), k}
        # where p_k is the probability of the bin containing (i,j)
        
        # For efficiency, we can compute the gradient contribution per bin
        grad = np.zeros_like(P)
        for b in range(num_bins):
            if counts[b] > 1e-12:
                p_b = counts[b] / total
                # dH/db = -1/ln2 * (log2(p_b) + 1)
                # dH/dP_ij = (1/total) * dH/db for i,j in bin b
                coeff = -1.0 / (np.log(2.0) * total) * (np.log2(p_b) + 1.0)
                mask = (mapping == b)
                grad[mask] += coeff
        
        # Wait, the derivative of H = -sum p_i log p_i w.r.t P_j (where P_j contributes to p_k)
        # p_k = sum_{i in bin k} P_i / Total
        # H = - sum_k p_k log2 p_k
        # dH/dP_j = - sum_k (d p_k / d P_j) (log2 p_k + 1/ln 2)
        # d p_k / d P_j = 1/Total if mapping(j)=k else 0
        # So dH/dP_j = - (1/Total) (log2 p_{mapping(j)} + 1/ln 2)
        
        # My previous derivation:
        # coeff = -1/(ln 2 * Total) * (log2(p_b) + 1)
        # This matches.
        
        # Calculate entropy value
        non_zero = counts > 1e-12
        H = -np.sum(probs[non_zero] * np.log2(probs[non_zero]))
        
        return H, grad

    # x - y
    map_m1 = (x_idx - y_idx + 9)
    H_m1, grad_m1 = entropy_and_grad(map_m1, 19)
    
    # x
    map_x = x_idx
    H_x, grad_x = entropy_and_grad(map_x, 10)
    
    # x + y
    map_p1 = (x_idx + y_idx)
    H_p1, grad_p1 = entropy_and_grad(map_p1, 19)
    
    # x + 2y
    map_p2 = (x_idx + 2 * y_idx)
    H_p2, grad_p2 = entropy_and_grad(map_p2, 28)
    
    M = max(H_x, H_p1, H_p2)
    
    if M < 1e-9:
        return 0.0, np.zeros_like(x_flat)
    
    ratio = H_m1 / M
    value = -ratio
    
    # Gradient of -ratio = -(1/M * grad H_m1 - H_m1/M^2 * grad M)
    # grad M = grad of max(H_x, H_p1, H_p2).
    # The gradient of max is the gradient of the active term(s).
    # If there is a tie, it's the average? No, for optimization we can pick the max one,
    # or if we want smoothness, we might need a smooth max.
    # For now, let's use the gradient of the term that achieved the max.
    # If there are multiple, we can average them or just pick one. Picking one is standard for max.
    # However, if H_x == H_p1 == H_p2, the gradient is undefined.
    # Let's check which one is max.
    
    active_grad = np.zeros_like(P)
    max_val = -1
    if H_x >= H_p1 and H_x >= H_p2:
        active_grad = grad_x
        max_val = H_x
    elif H_p1 >= H_x and H_p1 >= H_p2:
        active_grad = grad_p1
        max_val = H_p1
    else:
        active_grad = grad_p2
        max_val = H_p2
        
    # To handle ties more gracefully, we could weight by softmax of entropies,
    # but simple max is often sufficient for L-BFGS-B to find a local optimum.
    # Let's try simple max first.
    
    grad_ratio = (1.0/M) * grad_m1 - (H_m1 / (M*M)) * active_grad
    
    # We are minimizing -ratio, so gradient is -grad_ratio
    grad_obj = -grad_ratio
    
    return value, grad_obj.flatten()

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M using gradient-based optimization.
    """
    start_time = time.time()
    deadline = start_time + 590.0

    n = 10
    
    # Initial guess: Uniform distribution
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    x0 = P0.flatten()
    
    best_P = P0.copy()
    best_score = _fast_ratio(best_P)
    
    # Bounds: [0, 1] for each element. Sum constraint is handled by normalization in objective,
    # but L-BFGS-B works on box constraints.
    # Since we normalize inside, the sum constraint is not strictly enforced by the optimizer's bounds,
    # but the objective is scale-invariant (ratio of entropies).
    # Actually, if we normalize inside, the gradient calculation assumes P sums to 1.
    # If we let the optimizer vary the sum, the gradient w.r.t the raw x_flat is different.
    # It is safer to optimize on the simplex directly or use a parameterization.
    # However, L-BFGS-B with bounds [0,1] and normalization inside is a common "hack" that works
    # if the normalization is smooth. But the gradient I derived assumed P sums to 1.
    # If P does not sum to 1, the gradient is wrong.
    
    # Alternative: Use a softmax parameterization or just project onto simplex.
    # Given time, let's try a simpler approach:
    # Optimize log-weights? Or just use the normalized gradient?
    
    # Let's refine the gradient calculation to account for normalization.
    # Let P = x / sum(x).
    # f(x) = -H(x/sum(x)) / M(x/sum(x))
    # This is complex.
    
    # Simpler strategy:
    # Use scipy.optimize.minimize with method='SLSQP' which can handle equality constraints (sum=1).
    # Or use 'L-BFGS-B' on a transformed space.
    
    # Let's stick to the current gradient but assume we are on the simplex.
    # We can enforce the sum constraint by projecting the step back onto the simplex.
    # L-BFGS-B does not support custom projections.
    
    # Let's try SLSQP with constraint sum(P)=1.
    
    def objective_slsqp(x_flat):
        P = x_flat.reshape(10, 10)
        P = np.maximum(P, 0)
        s = P.sum()
        if s < 1e-12:
            return 0.0
        P = P / s
        return -_fast_ratio(P)
        
    # SLSQP is slower but handles constraints.
    # However, for 100 variables, SLSQP might be slow.
    
    # Let's go back to L-BFGS-B and just trust that the normalization inside the objective
    # combined with the gradient derived for normalized P will work "well enough"
    # because the optimizer will quickly converge to a region where sum(P) is stable.
    # Actually, if I pass x_flat that doesn't sum to 1, the gradient I return is wrong.
    
    # Better approach:
    # Parameterize P using 99 free variables (drop the last one) and calculate the 100th as 1 - sum(others).
    # This ensures sum=1.
    # x_free: 99 dims.
    # P[0..98] = x_free
    # P[99] = 1 - sum(x_free)
    # Constraint: x_free[i] >= 0 and 1 - sum(x_free) >= 0.
    # This is a convex polytope (simplex).
    # L-BFGS-B works with box constraints.
    # Bounds for x_free: [0, 1].
    # But we need 1 - sum(x_free) >= 0. This is a linear constraint, not a box constraint.
    # L-BFGS-B cannot handle linear inequality constraints directly (only bounds).
    
    # So we must use SLSQP or COBYLA or a custom projection.
    
    # Let's use SLSQP. It might be slower but robust.
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(100)}]
    bounds = [(0.0, 1.0) for _ in range(100)]
    
    # Initial point
    x0_slsqp = P0.flatten()
    
    # Run SLSQP
    try:
        res = minimize(objective_slsqp, x0_slsqp, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
        if time.time() > deadline - 10:
            pass # Break out if needed, but SLSQP is one call.
        
        if res.success:
            P_opt = res.x.reshape(10, 10)
            P_opt = np.maximum(P_opt, 0)
            P_opt = P_opt / P_opt.sum()
            score_opt = _fast_ratio(P_opt)
            if score_opt > best_score:
                best_P = P_opt
                best_score = score_opt
    except Exception as e:
        pass # Fallback to initial
    
    # If SLSQP is too slow or fails, try a few random restarts with SLSQP?
    # Or try Nelder-Mead on a reduced set of parameters?
    
    # Let's add a local search polish on the best found so far using the fast ratio
    # to ensure we aren't stuck in a numerical precision trap.
    
    # Simple local polish:
    current_P = best_P.copy()
    current_score = best_score
    
    rng = np.random.RandomState(42)
    
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Perturb a random cell
        i, j = rng.randint(10), rng.randint(10)
        old_val = current_P[i, j]
        
        # Try moving weight
        for factor in [0.5, 0.8, 1.2, 1.5, 0.0]:
            new_P = current_P.copy()
            if factor == 0.0:
                new_P[i, j] = 0.0
            else:
                new_P[i, j] = old_val * factor
            
            # Renormalize
            s = new_P.sum()
            if s < 1e-9:
                continue
            new_P = new_P / s
            
            new_score = _fast_ratio(new_P)
            if new_score > current_score:
                current_P = new_P
                current_score = new_score
                if current_score > best_score:
                    best_P = current_P
                    best_score = current_score

    # Final cleanup
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((10, 10), dtype=np.float64) / 100.0

    return best_P

# EVOLVE-BLOCK-END
