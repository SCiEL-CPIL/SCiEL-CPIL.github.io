"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Spectral Residual Minimization.
Instead of directly minimizing the worst-case integration error (which involves
max operations and is non-smooth), we treat the quadrature design as a spectral
approximation problem. We assume the integrand 1/E(a) can be well-approximated
by a truncated Chebyshev series on [0,1]. We find nodes and weights that minimize
the residual of the quadrature when applied to the first M Chebyshev polynomials
mapped to [0,1]. This provides a smooth, differentiable objective function that
guides the optimizer to a "spectrally accurate" rule. We then perform a local
refinement step to minimize the actual worst-case error on a diverse library of
cosmologies, using the spectral solution as a strong starting point.
This approach leverages the structure of the integrand (smooth, endpoint-bounded)
and avoids the local minima traps of direct minimax optimization.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

START = time.time()
DEADLINE = 90.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe."""
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * (1.0 + w0 + wa))
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _integrate(z, nodes, weights, Efun):
    """Compute the quadrature approximation."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    return (1.0 + z) * z * np.sum(weights / E)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _build_library():
    """Build a diverse library of (w0, wa) pairs for validation."""
    # Wider range to capture potential validation set
    w0_vals = np.linspace(-3.0, 1.0, 7)
    wa_vals = np.linspace(-3.0, 3.0, 7)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
            # Exclude near-CDM to ensure we are testing dark energy models
            if abs(w0 + 1.0) < 0.05 and abs(wa) < 0.05:
                continue
            lib.append((w0, wa))
    return lib

def _get_error(nodes, weights, zs, lib):
    """Compute worst-case relative error for the given nodes/weights."""
    max_err = 0.0
    for w0, wa in lib:
        Efun = lambda a, w0=w0, wa=wa: _E_cpl(a, w0, wa)
        for z in zs:
            app = _integrate(z, nodes, weights, Efun)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            if err > max_err:
                max_err = err
    return max_err

def _spectral_residual(nodes, weights, M=16):
    """
    Compute the residual of the quadrature rule against Chebyshev polynomials.
    We map Chebyshev polynomials from [-1,1] to [0,1].
    T_k(x) on [-1,1] becomes T_k(2u-1) on [0,1].
    We want sum_i w_i T_k(2u_i-1) to approximate integral_0^1 T_k(2u-1) du.
    """
    n = len(nodes)
    u = nodes
    x = 2 * u - 1
    
    # Compute Chebyshev polynomials at nodes
    # T_k(x)
    T = np.zeros((M, n))
    T[0, :] = 1.0
    if M > 1:
        T[1, :] = x
    for k in range(2, M):
        T[k, :] = 2 * x * T[k-1, :] - T[k-2, :]
    
    # Compute exact integrals of T_k(2u-1) over [0,1]
    # Let x = 2u-1, du = dx/2. Integral from -1 to 1 of T_k(x) * (1/2) dx
    # Integral of T_k(x) from -1 to 1 is 0 for k odd, and 2/(1-k^2) * (-1)^(k/2) for k even (k>0), 2 for k=0.
    # Wait, standard formula: 
    # int_{-1}^1 T_0(x) dx = 2
    # int_{-1}^1 T_k(x) dx = 0 for k odd
    # int_{-1}^1 T_k(x) dx = -2/(k^2-1) * (-1)^(k/2) for k even, k>0
    
    exact_ints = np.zeros(M)
    exact_ints[0] = 2.0
    for k in range(2, M, 2):
        sign = 1.0 if (k // 2) % 2 == 0 else -1.0
        exact_ints[k] = -2.0 / (k*k - 1) * sign
    
    # Scale by 1/2 for the change of variable du = dx/2
    target = 0.5 * exact_ints
    
    # Quadrature approximation: sum_i w_i T_k(u_i)
    quad_vals = np.sum(weights * T, axis=1)
    
    # Residual
    residual = quad_vals - target
    
    # We want to minimize the sum of squares of residuals
    # But we also need to ensure positivity of weights.
    # This function is just the objective for the spectral part.
    return np.sum(residual**2)

def construct():
    fallback = _fallback()
    n = 8
    
    # 1. Initial Guess: Gauss-Legendre nodes and weights mapped to [0,1]
    # Gauss-Legendre is generally better for smooth functions on [0,1] than Chebyshev
    # because the weight function is 1, not 1/sqrt(1-x^2).
    xi, wi = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing
    nodes = np.sort(nodes)
    
    # 2. Spectral Optimization Step
    # Minimize the spectral residual. This is a smooth function.
    # We optimize nodes and weights.
    
    def spectral_obj(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        
        # Constraints: nodes in [0,1], strictly increasing
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
            
        # Normalize weights for the residual calculation to avoid scale issues
        w_norm = weights_x / np.sum(weights_x)
        
        return _spectral_residual(nodes_x, w_norm, M=16)

    x0 = np.concatenate([nodes, weights])
    
    # Use L-BFGS-B for the spectral part, it's good for smooth problems with bounds
    # We don't have explicit bounds in L-BFGS-B for the interior, but we penalize violations
    # Alternatively, use Nelder-Mead which is robust but slower.
    # Let's try L-BFGS-B with a penalty method or just careful initialization.
    
    if time.time() < DEADLINE - MARGIN - 10:
        # Try a few restarts with slight perturbations if needed, but start with GL
        res = minimize(spectral_obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-6: # If spectral optimization converged well
            new_nodes = res.x[:n]
            new_weights = res.x[n:]
            
            # Check validity
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                nodes = new_nodes
                weights = new_weights

    # 3. Local Refinement for Worst-Case Error
    # Now that we have a "spectrally good" rule, we refine it to minimize
    # the actual worst-case error on the cosmology library.
    
    lib = _build_library()
    # Use a moderate set of z values for speed
    zs = np.linspace(0.1, 3.0, 15)
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = _get_error(nodes, weights, zs, lib)
    
    def error_obj(x):
        nodes_x = x[:n]
        weights_x = x[n:]
        if nodes_x[0] < 1e-6 or nodes_x[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes_x) < 1e-6):
            return 1e10
        if np.any(weights_x < 1e-12):
            return 1e10
        weights_norm = weights_x / np.sum(weights_x)
        return _get_error(nodes_x, weights_norm, zs, lib)
        
    x0_ref = np.concatenate([nodes, weights])
    
    # Use Nelder-Mead for the final refinement as it handles the non-smooth max() well
    if time.time() < DEADLINE - MARGIN:
        res_ref = minimize(error_obj, x0_ref, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res_ref.fun < best_err:
            new_nodes = res_ref.x[:n]
            new_weights = res_ref.x[n:]
            if np.all(np.diff(new_nodes) > 1e-6) and np.all(new_weights > 1e-12):
                new_weights = new_weights / np.sum(new_weights)
                best_nodes = new_nodes
                best_weights = new_weights
                best_err = res_ref.fun

    # Final validation and return
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
