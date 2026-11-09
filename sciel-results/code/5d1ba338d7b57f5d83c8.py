"""Refined Cohn-Goncalves Laguerre-polynomial construction."""
import time
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_c4_bound(root_positions, m=12, alpha=1.0, n_quad=200):
    """
    Compute the C4 upper bound given double-root positions.
    The bound involves the 4th moment ratio of the polynomial
    P(x) = prod_i (x - r_i)^2 under the Laguerre weight x^alpha e^{-x}.
    
    C4 ~ (integral of P(x)^2 * x^4 * x^alpha * e^{-x} dx) /
         (integral of P(x) * x^alpha * e^{-x} dx)^2 * normalization
    """
    if len(root_positions) != m:
        return np.inf
    
    # Use Gauss-Laguerre quadrature for the integral
    x_nodes, w_nodes = roots_genlaguerre(n_quad, alpha)
    
    # Compute P(x) = prod_i (x - r_i)^2 at quadrature nodes
    # Work in log space for stability
    log_P = np.zeros_like(x_nodes)
    for r in root_positions:
        log_P += 2.0 * np.log(np.abs(x_nodes - r) + 1e-300)
    
    # P(x) = exp(log_P)
    P = np.exp(log_P)
    
    # Weight function: x^alpha * e^{-x} is already in w_nodes
    # integral of f(x) * x^alpha * e^{-x} dx ≈ sum w_i * f(x_i)
    
    # Numerator: integral of P(x)^2 * x^4 * weight
    numerator = np.sum(w_nodes * P**2 * x_nodes**4)
    
    # Denominator: (integral of P(x) * weight)^2
    denom = np.sum(w_nodes * P)
    denominator = denom**2
    
    if denominator <= 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.inf
    
    return numerator / denominator


def _compute_c4_bound_direct(root_positions, m=12, alpha=1.0, n_quad=200):
    """
    Alternative computation using direct polynomial evaluation.
    For the Fourier sign uncertainty, the C4 bound is related to
    the ratio of the 4th moment to the square of the 0th moment
    of the squared polynomial under the Laguerre weight.
    """
    if len(root_positions) != m:
        return np.inf
    
    x_nodes, w_nodes = roots_genlaguerre(n_quad, alpha)
    
    # Build P(x) = prod (x - r_i)^2
    # For numerical stability, compute in chunks
    P = np.ones_like(x_nodes)
    for r in root_positions:
        P *= (x_nodes - r)**2
    
    # Check for overflow/underflow
    if not np.all(np.isfinite(P)):
        return np.inf
    
    numerator = np.sum(w_nodes * P**2 * x_nodes**4)
    denom = np.sum(w_nodes * P)
    denominator = denom**2
    
    if denominator <= 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.inf
    
    return numerator / denominator


def _objective(params, m=12, alpha=1.0, n_quad=100):
    """Objective function for optimization: minimize C4 bound."""
    # params are the 12 root positions, must be positive
    if np.any(params <= 0):
        return 1e10 + np.sum((params - 1.0)**2)
    
    try:
        c4 = _compute_c4_bound_direct(params, m, alpha, n_quad)
        if not np.isfinite(c4):
            return 1e10
        return c4
    except Exception:
        return 1e10


def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    Uses optimization to minimize the C4 upper bound.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0
    
    m = 12
    alpha = 1.0
    
    # Initial guess: roots of L_12^(1)(x)
    try:
        x0, _ = roots_genlaguerre(m, alpha)
        x0 = sorted(x0)
    except Exception:
        x0 = np.linspace(0.5, 50.0, m)
    
    best_roots = list(x0)
    best_score = _compute_c4_bound_direct(best_roots, m, alpha, 100)
    
    if not np.isfinite(best_score):
        best_score = 1e10
    
    # Try optimization with different starting points
    n_quad_opt = 80  # Fewer quadrature points for speed during optimization
    
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        if time.time() + safety_margin > deadline:
            break
        
        # Different initializations
        if attempts == 0:
            x_init = x0.copy()
        elif attempts == 1:
            # Perturb the initial roots
            x_init = x0 + 0.1 * np.random.randn(m)
            x_init = np.maximum(x_init, 0.1)
        elif attempts == 2:
            # Spread out more
            x_init = np.linspace(0.5, 60.0, m)
        elif attempts == 3:
            # Cluster more
            x_init = np.linspace(0.3, 30.0, m)
        else:
            x_init = x0 * np.exp(0.2 * np.random.randn(m))
            x_init = np.maximum(x_init, 0.1)
        
        attempts += 1
        
        # Constrain: roots must be positive and ordered
        # Use log-space parameterization: r_i = exp(y_i)
        y_init = np.log(np.maximum(x_init, 1e-6))
        
        def _obj_log(y):
            r = np.exp(y)
            return _objective(r, m, alpha, n_quad_opt)
        
        try:
            result = minimize(
                _obj_log,
                y_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                r_opt = np.exp(result.x)
                if np.all(r_opt > 0):
                    # Sort to ensure consistent ordering
                    r_opt = sorted(r_opt)
                    score = _compute_c4_bound_direct(r_opt, m, alpha, 100)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_roots = list(r_opt)
        except Exception:
            pass
        
        if time.time() + safety_margin > deadline:
            break
    
    # Final refinement with more quadrature points
    if time.time() + safety_margin > deadline:
        return [float(r) for r in best_roots]
    
    # One more round of refinement with higher precision quadrature
    try:
        y_init = np.log(np.maximum(best_roots, 1e-6))
        
        def _obj_high(y):
            r = np.exp(y)
            return _objective(r, m, alpha, 150)
        
        result = minimize(
            _obj_high,
            y_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.fun < best_score:
            r_final = np.exp(result.x)
            if np.all(r_final > 0):
                r_final = sorted(r_final)
                score = _compute_c4_bound_direct(r_final, m, alpha, 200)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_roots = list(r_final)
    except Exception:
        pass
    
    # Verify constraints
    if len(best_roots) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    if any(r <= 0 for r in best_roots):
        return [4.0 * (i + 1) for i in range(12)]
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-END
