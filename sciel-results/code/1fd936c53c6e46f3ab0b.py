"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, eval_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 570.0  # 30s safety margin before 600s kill
M = 12
N = 1

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_objective(roots):
    """
    Compute the surrogate for the C4 upper bound.
    We use the ratio of the L2 norm to the square of the L1 norm of the 
    double-root Laguerre combination P(x) = prod (x - r_i)^2 under the weight e^{-x}.
    This ratio is a measure of the 'concentration' or 'spread' of the polynomial,
    which correlates with the tightness of the uncertainty bound.
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or not np.all(np.diff(roots) > 0):
        return 1e10

    # Use Gauss-Laguerre quadrature nodes and weights for high accuracy
    # 200 nodes is sufficient for smooth polynomials of degree 24
    x_nodes, w_nodes = np.polynomial.laguerre.laggauss(200)
    
    # Compute P(x) = prod (x - r_i)^2
    # To avoid overflow/underflow, compute in log-space if possible, 
    # but for 12 roots and moderate x, direct multiplication is usually fine.
    # However, P(x) can be large. Let's normalize by subtracting the mean root?
    # No, the weight e^{-x} handles the tail.
    
    # Efficient evaluation: 
    # P(x) = (L_12^{(1)}(x))^2 is the initial guess, but we are optimizing roots.
    # So we must compute the product.
    
    # Vectorized product
    # x_nodes shape: (200,)
    # roots shape: (12,)
    # diff shape: (200, 12)
    diffs = x_nodes[:, None] - roots[None, :]
    # P(x) = prod_i (diffs_ij)^2
    # log_P = sum_j 2 * log(|diffs_ij|)
    # We need to handle x == r_i (P=0). 
    # In quadrature, nodes rarely hit roots exactly, but we should be careful.
    # Using log-sum-exp for stability?
    # log_P = 2 * sum(log(|x - r|))
    
    # To avoid log(0), we can add a tiny epsilon, but roots are fixed parameters here.
    # If a node hits a root, P=0, contribution is 0.
    
    abs_diffs = np.abs(diffs)
    # Mask where diff is 0
    mask = abs_diffs < 1e-15
    abs_diffs[mask] = 1.0 # log(1)=0, so contribution to log sum is 0, but P is 0.
    # Actually, if P=0, the term is 0. 
    # Let's compute P directly. For 12 roots, product is 12th power.
    # If roots are around 1-50, x_nodes go up to ~400.
    # (400)^24 is huge. 
    # We should compute the objective in a scaled way or use logs.
    
    # Let's compute log(P(x))
    # log_P = 2 * sum(log(|x - r|))
    # Where diff is 0, log_P -> -inf, so P=0.
    # We can compute exp(log_P) * w.
    # To prevent overflow in exp, we can subtract the max log_P.
    
    log_abs_diffs = np.log(abs_diffs)
    log_P = 2.0 * np.sum(log_abs_diffs, axis=1)
    
    # Handle the mask where P should be 0
    log_P[mask.any(axis=1)] = -np.inf
    
    # We want to minimize J = ( sum w * P^2 ) / ( sum w * P )^2
    # Let A = sum w * P^2, B = sum w * P
    # J = A / B^2
    
    # Compute A and B using log-sum-exp
    # A = sum w_i exp(2 * log_P_i)
    # B = sum w_i exp(log_P_i)
    
    # Max for stability
    max_log_P = np.max(log_P)
    if np.isneginf(max_log_P):
        return 1e10 # All P are 0? Unlikely.
        
    exp_log_P = np.exp(log_P - max_log_P)
    
    # B_scaled = sum w * exp_log_P
    B_scaled = np.sum(w_nodes * exp_log_P)
    
    # A_scaled = sum w * exp(2 * (log_P - max_log_P)) = sum w * exp_log_P^2
    A_scaled = np.sum(w_nodes * exp_log_P**2)
    
    # J = (A_scaled * exp(2*max_log_P)) / (B_scaled * exp(max_log_P))^2
    # J = (A_scaled * exp(2*max_log_P)) / (B_scaled^2 * exp(2*max_log_P))
    # J = A_scaled / B_scaled^2
    
    # The max_log_P cancels out! This is great for numerical stability.
    
    if B_scaled == 0:
        return 1e10
        
    obj = A_scaled / (B_scaled * B_scaled)
    
    # Add a penalty for roots being too close or too far
    gaps = np.diff(roots)
    min_gap = np.min(gaps)
    if min_gap < 1e-6:
        obj += 1e6 / min_gap
        
    return obj

def construct():
    start_time = time.time()
    
    # 1. Initial Guess: Zeros of L_12^(1)
    # roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha)
    initial_roots = roots_genlaguerre(M, 2*N - 1) # alpha = 1
    initial_roots = np.sort(initial_roots)
    
    # Ensure positive
    initial_roots = np.maximum(initial_roots, 1e-6)
    
    best_roots = initial_roots.copy()
    best_obj = _compute_objective(best_roots)
    
    # 2. Optimization
    # Use L-BFGS-B with bounds
    bounds = [(1e-6, None)] * M
    
    # We need to ensure ordering r_1 < r_2 < ... < r_12
    # L-BFGS-B does not handle inequality constraints between variables directly.
    # We can parameterize using gaps: r_i = r_1 + sum_{j=1}^{i-1} gap_j
    # Variables: [r_1, gap_1, ..., gap_11]
    # r_1 > 0, gap_i > 0
    
    def objective_gaps(params):
        if _deadline_reached():
            return 1e10
        r1 = params[0]
        gaps = params[1:]
        if r1 <= 0 or np.any(gaps <= 0):
            return 1e10
        roots = np.empty(M)
        roots[0] = r1
        roots[1:] = r1 + np.cumsum(gaps)
        return _compute_objective(roots)
    
    # Initial gaps
    r1 = initial_roots[0]
    gaps = np.diff(initial_roots)
    init_params = np.concatenate(([r1], gaps))
    
    # Optimize
    try:
        res = minimize(
            objective_gaps,
            init_params,
            method='L-BFGS-B',
            bounds=[(1e-6, None)] + [(1e-6, None)] * (M - 1),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            r1 = res.x[0]
            gaps = res.x[1:]
            best_roots = np.empty(M)
            best_roots[0] = r1
            best_roots[1:] = r1 + np.cumsum(gaps)
    except Exception:
        pass
    
    # 3. Polish with tighter tolerance if time remains
    if not _deadline_reached():
        try:
            res2 = minimize(
                objective_gaps,
                np.concatenate(([best_roots[0]], np.diff(best_roots))),
                method='L-BFGS-B',
                bounds=[(1e-6, None)] + [(1e-6, None)] * (M - 1),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res2.success and res2.fun < best_obj:
                best_obj = res2.fun
                r1 = res2.x[0]
                gaps = res2.x[1:]
                best_roots = np.empty(M)
                best_roots[0] = r1
                best_roots[1:] = r1 + np.cumsum(gaps)
        except Exception:
            pass

    # Final safety check
    if _deadline_reached():
        return [float(x) for x in initial_roots]
    
    # Ensure valid output
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(x) for x in best_roots]
# EVOLVE-BLOCK-END
