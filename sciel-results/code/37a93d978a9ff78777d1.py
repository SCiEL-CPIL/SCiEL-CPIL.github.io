"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _cpl_E(z, w0, w1):
    """Compute E(z) = H(z)/H0 for CPL parametrization."""
    a = 1.0 / (1.0 + z)
    return (1.0 - a) ** (3.0 * (1.0 + w0 + w1)) * a ** (-3.0 * w1)


def _integrand(z, w0, w1):
    """Integrand for luminosity distance: z / E(z)."""
    if z == 0:
        return 0.0
    return z / _cpl_E(z, w0, w1)


def _true_integral(z_max, w0, w1, n_pts=4000):
    """Compute reference integral using dense adaptive-like sampling."""
    z = np.linspace(0, z_max, n_pts)
    f = np.array([_integrand(zi, w0, w1) for zi in z])
    return np.trapz(f, z)


def _quad_value(nodes, weights, z_max, w0, w1):
    """Compute quadrature approximation of the integral."""
    z_nodes = z_max * nodes
    f_vals = np.array([_integrand(zi, w0, w1) for zi in z_nodes])
    return np.sum(weights * f_vals)


def _objective(params, test_cases):
    """Minimax objective: maximize the minimum of (1 - |rel_error|) across cases.
    We minimize the max relative error."""
    n = 8
    # Unpack and constrain
    nodes = params[:n]
    log_weights = params[n:]
    
    # Enforce ordering with soft penalty
    diffs = np.diff(nodes)
    if np.any(diffs <= 0):
        return 1e10
    if nodes[0] < 0 or nodes[-1] > 1:
        return 1e10
    
    weights = np.exp(log_weights)
    w_sum = np.sum(weights)
    weights = weights / w_sum  # normalize
    
    # Compute max relative error
    max_err = 0.0
    for z_max, w0, w1 in test_cases:
        ref = _true_integral(z_max, w0, w1)
        if abs(ref) < 1e-12:
            continue
        quad = _quad_value(nodes, weights, z_max, w0, w1)
        rel_err = abs(quad - ref) / abs(ref)
        if rel_err > max_err:
            max_err = rel_err
    
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin before 90s kill
    
    # Test cases: (z_max, w0, w1) covering a range of CPL models
    test_cases = [
        (0.5, -1.0, 0.0),    # Cosmological constant
        (1.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
        (0.5, -0.9, -0.3),   # Phantom-like
        (1.0, -0.9, -0.3),
        (2.0, -0.9, -0.3),
        (0.5, -0.5, -0.5),   # Quintessence
        (1.0, -0.5, -0.5),
        (2.0, -0.5, -0.5),
        (0.5, -1.0, 0.5),    # WMAP-like
        (1.0, -1.0, 0.5),
        (2.0, -1.0, 0.5),
        (0.5, -0.7, -0.2),   # Intermediate
        (1.0, -0.7, -0.2),
        (2.0, -0.7, -0.2),
        (0.3, -1.0, 0.0),
        (1.5, -1.0, 0.0),
        (0.3, -0.5, -0.5),
        (1.5, -0.5, -0.5),
    ]
    
    # Initial guess: Gauss-Legendre nodes on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (x_gl + 1.0)  # map [-1,1] to [0,1]
    weights_init = 0.5 * w_gl
    log_weights_init = np.log(weights_init)
    
    params_init = np.concatenate([nodes_init, log_weights_init])
    
    best_params = params_init.copy()
    best_err = _objective(best_params, test_cases)
    
    # Try multiple random restarts with optimization
    rng = np.random.RandomState(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            p0 = params_init.copy()
        else:
            # Perturb around best or random
            if restart < 3:
                p0 = best_params + rng.randn(16) * 0.05
            else:
                # More random: spread nodes
                nodes_rand = np.sort(rng.uniform(0.01, 0.99, 8))
                w_rand = np.abs(rng.randn(8)) + 0.1
                w_rand /= w_rand.sum()
                p0 = np.concatenate([nodes_rand, np.log(w_rand)])
        
        result = minimize(
            _objective,
            p0,
            args=(test_cases,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x.copy()
        
        # Also try L-BFGS-B with bounds for a different landscape
        if time.time() > deadline:
            break
        
        # Convert to bounded form for L-BFGS-B
        def _bounded_obj(bparams):
            # bparams: 8 nodes in (0,1) sorted, 8 log-weights
            bnodes = bparams[:8]
            blogw = bparams[8:]
            
            # Enforce ordering: use cumulative softplus
            # Actually, just check
            if np.any(np.diff(bnodes) <= 0):
                return 1e10
            if bnodes[0] < 0 or bnodes[-1] > 1:
                return 1e10
            
            bw = np.exp(blogw)
            bw /= bw.sum()
            
            max_err = 0.0
            for z_max, w0, w1 in test_cases:
                ref = _true_integral(z_max, w0, w1)
                if abs(ref) < 1e-12:
                    continue
                z_nodes = z_max * bnodes
                f_vals = np.array([_integrand(zi, w0, w1) for zi in z_nodes])
                quad = np.sum(bw * f_vals)
                rel_err = abs(quad - ref) / abs(ref)
                if rel_err > max_err:
                    max_err = rel_err
            return max_err
        
        # Get current best as starting point
        bnodes_best = best_params[:8]
        blogw_best = best_params[8:]
        b0 = np.concatenate([bnodes_best, blogw_best])
        
        bounds = [(0.001, 0.999)] * 8 + [(-10, 10)] * 8
        
        result2 = minimize(
            _bounded_obj,
            b0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result2.fun < best_err:
            best_err = result2.fun
            best_params = result2.x.copy()
    
    # Extract final result
    nodes = best_params[:8].copy()
    log_weights = best_params[8:].copy()
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    
    # Ensure strictly increasing
    nodes = np.sort(nodes)
    
    # Final validation
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0)
    assert nodes[0] >= 0 and nodes[-1] <= 1
    assert np.all(weights > 0)
    assert abs(weights.sum() - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
