import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, roots_laguerre
import time

def _compute_c4_fast(roots):
    """
    Compute the C4 upper bound using Gauss-Laguerre quadrature.
    C4 = 1 / sqrt( integral_0^inf P(x)^2 * (L_m^(0)(x))^2 * exp(-x) dx )
    We want to minimize C4, which means maximizing the integral.
    """
    m = 12
    # Get Gauss-Laguerre nodes and weights for N=50 (exact for deg < 99)
    # The integrand polynomial degree is 2*deg(P) + 2*deg(L_m) = 2*24 + 2*12 = 72.
    # N=40 is sufficient (2*40-1 = 79 > 72). Use N=50 for safety.
    nodes, weights = roots_laguerre(50)
    
    # Evaluate P(x) = prod (x - r_i)^2 at nodes
    # Vectorized computation
    # nodes shape (N,), roots shape (m,)
    # diff shape (N, m)
    diff = nodes[:, None] - roots[None, :]
    # P_val = prod(diff^2, axis=1)
    # To avoid overflow/underflow issues, we can sum logs, but values are likely moderate.
    # Let's just compute directly.
    P_vals = np.prod(diff**2, axis=1)
    
    # Evaluate Laguerre polynomial L_12^(0)(x) at nodes
    L_vals = eval_genlaguerre(m, 0, nodes)
    
    # Integrand: P(x)^2 * L(x)^2 * exp(-x)
    # Quadrature: sum( weights * integrand )
    # Note: roots_laguerre weights already include the exp(-x) factor?
    # scipy.special.roots_laguerre returns nodes x_i and weights w_i such that
    # integral_0^inf f(x) exp(-x) dx ~ sum w_i f(x_i)
    # So we compute sum( weights * (P_vals * L_vals)**2 )
    
    integrand_vals = (P_vals * L_vals)**2
    integral_val = np.sum(weights * integrand_vals)
    
    if integral_val <= 0:
        return np.inf
        
    return 1.0 / np.sqrt(integral_val)

def _objective(roots):
    """
    Objective function for minimization.
    We want to minimize C4.
    """
    try:
        c4 = _compute_c4_fast(roots)
        if not np.isfinite(c4):
            return 1e10
        return c4
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    m = 12
    
    # Define bounds for differential evolution
    # Roots should be positive.
    # Typical scale for Laguerre roots is O(m).
    # Let's search in [0.1, 50.0]
    bounds = [(0.1, 50.0)] * m
    
    best_roots = None
    best_c4 = np.inf
    
    # Phase 1: Global Search using Differential Evolution
    # DE is robust for global optimization
    try:
        # tol=1e-6, maxiter=200
        # seed for reproducibility
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=200,
            tol=1e-6,
            polish=False, # We will polish manually
            update_interval=10
        )
        if res.fun < best_c4:
            best_c4 = res.fun
            best_roots = res.x
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return [float(r) for r in best_roots]
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random restarts if DE fails
        for _ in range(10):
            x0 = np.random.uniform(0.5, 20.0, m)
            c4 = _objective(x0)
            if c4 < best_c4:
                best_c4 = c4
                best_roots = x0

    if best_roots is None:
        # Fallback initial guess
        best_roots = np.linspace(1.0, 15.0, m)
        best_c4 = _objective(best_roots)

    # Phase 2: Local Refinement
    # Use L-BFGS-B or Powell to refine the best solution found by DE
    # L-BFGS-B is faster if we have gradients, but we don't. Powell is robust.
    # Let's try a few local starts from the DE result and some random perturbations
    
    local_starts = [best_roots.copy()]
    
    # Add some random perturbations of the best root
    for _ in range(5):
        perturbed = best_roots * (1 + np.random.normal(0, 0.1, m))
        perturbed = np.clip(perturbed, 0.1, 50.0)
        local_starts.append(perturbed)
        
    # Also try a few independent random starts with local optimization
    for _ in range(5):
        x0 = np.random.uniform(1.0, 20.0, m)
        local_starts.append(x0)

    for x0 in local_starts:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        try:
            res_local = minimize(
                _objective, 
                x0, 
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-8, 'disp': False}
            )
            if res_local.fun < best_c4:
                best_c4 = res_local.fun
                best_roots = res_local.x
        except:
            pass

    # Final check
    if best_roots is None:
        best_roots = np.linspace(1.0, 15.0, m)
        
    return [float(r) for r in best_roots]
