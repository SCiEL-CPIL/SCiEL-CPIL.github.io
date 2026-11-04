"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    # c = A_T_inv @ p where p is standard-basis coeffs (ascending)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute squared Laguerre norm of P(x) = prod(x - r_i).
        """
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize upper bound on C4 ~ 1/||P||^2.
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def cost_with_grad(log_roots):
        """
        Cost and analytic gradient.
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(log_roots)
        
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(log_roots)
        
        cost = 1.0 / norm_sq
        grad = np.zeros(12)
        
        # Precompute weights for gradient
        w = (np.arange(13) + 1)
        
        for i in range(12):
            # dP/dr_i = -prod_{j!=i}(x - r_j)
            other_roots = np.delete(roots, i)
            coeffs_other_desc = np.poly(other_roots)
            p_other = coeffs_other_desc[::-1]
            dp_dr_i = -p_other
            dc_dr_i = A_T_inv @ dp_dr_i
            dnorm_sq_dr_i = 2.0 * np.sum(c * w * dc_dr_i)
            dcost_dr_i = -dnorm_sq_dr_i / (norm_sq * norm_sq)
            grad[i] = dcost_dr_i * roots[i]
        
        return cost, grad

    # Strategy:
    # 1. Multi-start local optimization with analytic gradients (L-BFGS-B)
    # 2. Global DE search for robustness
    # 3. Final high-precision polish
    
    # Initialize with a reasonable tridiagonal matrix
    # Start with roots ~ geometric sequence, then find tridiag that reproduces them
    def roots_to_tridiag(roots):
        """Find tridiagonal matrix with given eigenvalues (using Lanczos-like init)."""
        # Simple initialization: diagonal = mean, off-diagonal = spread
        diag = np.full(12, np.mean(roots))
        offdiag = np.full(11, np.std(roots) * 0.5)
        return np.concatenate([diag, offdiag])

    # Seed roots: geometric sequence
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    
    # Bounds for log_roots
    # Roots should be positive.
    # If roots are too small or too large, the norm might behave badly.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Multi-start local optimization from diverse initial points
    initial_points = []
    
    # Geometric sequence
    initial_points.append(seed_log_roots.copy())
    
    # Various arithmetic/geometric mixes
    for a in [2.0, 3.0, 5.0, 8.0]:
        for q in [1.5, 1.8, 2.0, 2.5, 3.0]:
            r = a * (q ** np.arange(12))
            if r.max() < 5000:
                initial_points.append(np.log(r))
    
    # Random initial points
    np.random.seed(42)
    for _ in range(10):
        r = np.exp(np.random.uniform(-1.0, 7.0, 12))
        initial_points.append(np.log(r))

    for init_lr in initial_points:
        if not time_ok():
            break
        try:
            result = minimize(
                cost_with_grad,
                init_lr,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 2: Global DE search for robustness
    # Use multiple seeds and settings to explore the landscape
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    for seed in seeds:
        if not time_ok():
            break
        try:
            result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=150,
                popsize=20,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if best_log_roots is not None and time_ok():
        try:
            polish_result = minimize(
                cost_with_grad,
                best_log_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
