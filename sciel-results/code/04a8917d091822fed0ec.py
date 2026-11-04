"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    
    Uses Jacobi matrix parameterization: roots are eigenvalues of a symmetric
    tridiagonal matrix with positive diagonal and off-diagonal elements, ensuring
    positive ordered roots while allowing flexible root distribution.
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def jacobi_to_roots(log_diag, log_offdiag):
        """
        Convert Jacobi matrix parameters to roots (eigenvalues).
        log_diag: array of 12 log-diagonal elements
        log_offdiag: array of 11 log-off-diagonal elements
        Returns sorted positive eigenvalues.
        """
        # Reconstruct diagonal and off-diagonal
        diag = np.exp(log_diag)
        offdiag = np.exp(log_offdiag)
        
        # Build symmetric tridiagonal matrix
        J = np.zeros((12, 12))
        for i in range(12):
            J[i, i] = diag[i]
        for i in range(11):
            J[i, i+1] = offdiag[i]
            J[i+1, i] = offdiag[i]
        
        # Compute eigenvalues (sorted ascending)
        eigenvalues = np.linalg.eigvalsh(J)
        
        # Ensure positivity
        if np.any(eigenvalues <= 0) or np.any(np.isnan(eigenvalues)):
            return None
        
        return eigenvalues

    def cost_func(params):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        
        params: [log_diag_0, ..., log_diag_11, log_offdiag_0, ..., log_offdiag_10]
        """
        log_diag = params[:12]
        log_offdiag = params[12:23]
        
        roots = jacobi_to_roots(log_diag, log_offdiag)
        if roots is None:
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 23D Jacobi parameters using Differential Evolution.
    # 2. Local refinement using L-BFGS-B.
    
    # Seed: geometric sequence a=4, q=2 -> roots 4, 8, 16, ...
    # Approximate Jacobi matrix for these roots
    # For equally spaced or geometric roots, we can estimate diagonal/off-diagonal
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    
    # Initialize Jacobi parameters from seed roots
    # Simple initialization: set diagonal to mean of roots, off-diagonal to spread
    mean_root = np.mean(seed_roots)
    spread = np.std(seed_roots)
    
    # Initial log parameters
    init_log_diag = np.full(12, math.log(mean_root))
    init_log_offdiag = np.full(11, math.log(spread / 2.0))
    init_params = np.concatenate([init_log_diag, init_log_offdiag])
    
    # Bounds for log-space parameters
    # log_diag: roots range from ~4 to ~16384, so log in [1, 10]
    # log_offdiag: spread from small to large, log in [-2, 8]
    bounds = [(-2.0, 10.0)] * 12 + [(-3.0, 8.0)] * 11
    
    best_roots = None
    best_val = 1e10
    
    # Evaluate seed
    seed_val = cost_func(init_params)
    if seed_val < best_val:
        best_val = seed_val
        seed_roots_jac = jacobi_to_roots(init_log_diag, init_log_offdiag)
        if seed_roots_jac is not None:
            best_roots = seed_roots_jac
    
    # Run Differential Evolution with reduced population for 23D
    # popsize=15, maxiter=100 should be feasible within time budget
    try:
        if time_ok():
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=150, 
                popsize=15, 
                tol=1e-8, 
                polish=True,  # L-BFGS-B polish
                init='sobol',
                updating='deferred'
            )
            
            if result.success or result.fun < best_val:
                best_val = result.fun
                log_diag = result.x[:12]
                log_offdiag = result.x[12:23]
                best_roots = jacobi_to_roots(log_diag, log_offdiag)
                
    except Exception as e:
        pass

    # Additional Local Polish from multiple starting points
    # Generate diverse starting points
    start_points = []
    
    # Seed-based
    start_points.append(init_params.copy())
    
    # Random perturbations around seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = init_params + rng.randn(23) * 0.5
        start_points.append(perturbed)
    
    # Different scales
    for scale in [0.5, 1.0, 2.0]:
        scaled_diag = init_log_diag + math.log(scale)
        scaled_offdiag = init_log_offdiag + math.log(scale)
        start_points.append(np.concatenate([scaled_diag, scaled_offdiag]))
    
    for sp in start_points:
        if not time_ok():
            break
        try:
            res = minimize(cost_func, sp, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                log_diag = res.x[:12]
                log_offdiag = res.x[12:23]
                best_roots = jacobi_to_roots(log_diag, log_offdiag)
        except Exception:
            continue

    # Final high-precision polish if time allows
    if best_roots is not None and time_ok():
        try:
            final_log_diag = np.log(np.maximum(best_roots, 1e-6))
            # Estimate off-diagonal from root spacing
            spacings = np.diff(np.sort(best_roots))
            final_log_offdiag = np.log(np.maximum(spacings / 2.0, 1e-6))
            final_params = np.concatenate([final_log_diag, final_log_offdiag])
            
            res = minimize(cost_func, final_params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                log_diag = res.x[:12]
                log_offdiag = res.x[12:23]
                best_roots = jacobi_to_roots(log_diag, log_offdiag)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback: use seed geometric sequence
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
