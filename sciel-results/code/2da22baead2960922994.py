import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    
    # Use Gauss-Laguerre quadrature
    # The integrals are over [0, infinity) with weight e^{-x}
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) at quadrature points
    # P(x) = prod (x - x_i)^2
    # Use log-space for stability if roots are large or many
    # Here 12 roots, so direct product might be okay, but log is safer
    log_P = np.zeros(n_quad)
    for r in roots:
        # Avoid log(0) if x_quad == r, but x_quad are positive and roots are positive.
        # If they match exactly, log(0) is -inf, exp is 0.
        # However, P(x) is 0 at roots.
        # We add a small epsilon to avoid numerical issues in log if needed, 
        # but strictly P(root)=0.
        # Using abs is fine since (x-r)^2 is positive.
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 590  # Safety margin
    
    # Initial guess: Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Define bounds for DE
    # Roots should be positive.
    # Laguerre roots for n=12 are roughly in [0.1, 50].
    # Let's use [0.01, 100]
    bounds = [(0.01, 100.0)] * 12
    
    # Phase 1: Differential Evolution
    # Use a time limit or maxiter.
    # DE can be slow. Let's try maxiter=50, popsize=15
    try:
        if time.time() < deadline - 200:
            de_result = differential_evolution(
                _compute_c4_bound,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                update_interval=1
            )
            if de_result.success and de_result.fun < best_score:
                best_score = de_result.fun
                best_roots = de_result.x.copy()
    except Exception:
        pass
    
    # Phase 2: Local Refinement
    # Refine from DE result and from original Laguerre roots
    candidates = [best_roots.copy(), x0.copy()]
    
    # Add a few random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random points in log-uniform distribution?
        # Or just uniform in [0.1, 50]
        rand_roots = rng.uniform(0.1, 50.0, 12)
        rand_roots = np.sort(rand_roots)
        candidates.append(rand_roots)
    
    bounds_local = [(1e-6, None)] * 12
    
    for x_init in candidates:
        if time.time() > deadline - 50:
            break
        
        try:
            # Use L-BFGS-B for speed and accuracy
            result = minimize(
                _compute_c4_bound,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_local,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                candidate = result.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    # Verify
                    score = _compute_c4_bound(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
