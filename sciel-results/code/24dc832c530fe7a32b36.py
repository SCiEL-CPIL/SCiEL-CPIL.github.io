import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def _E_squared(a, Omega_m, w0, w1):
    """E^2(a) for CPL dark energy model in flat universe."""
    a = np.maximum(a, 1e-15)
    Omega_de = 1.0 - Omega_m
    # Matter term
    matter = Omega_m * a**(-3)
    # Dark energy term: Omega_DE * a^{-3(1+w0+w1)} * exp(-3*w1*(1-a))
    w_eff = w0 + w1 * (1 - a)
    # Use the exact integral form
    de = Omega_de * a**(-3.0) * np.exp(-3.0 * w1 * (1.0 - a)) * a**(-3.0 * w0)
    # Simplify: a^{-3} * a^{-3w0} * exp(-3w1(1-a)) = a^{-3(1+w0)} * exp(-3w1(1-a))
    de = Omega_de * a**(-3.0 * (1.0 + w0)) * np.exp(-3.0 * w1 * (1.0 - a))
    return matter + de

def _E_of_a(a, Omega_m, w0, w1):
    return np.sqrt(_E_squared(a, Omega_m, w0, w1))

def _integrand(t, z, Omega_m, w0, w1):
    """f(t) = 1/E(a(t)) where a(t) = 1/(1+z*t)"""
    a = 1.0 / (1.0 + z * t)
    return 1.0 / _E_of_a(a, Omega_m, w0, w1)

def _exact_dL(z, Omega_m, w0, w1):
    """Exact dL/(c/H0) = (1+z)*z * ∫_0^1 dt/E(a(t))"""
    if z <= 0:
        return 0.0
    val, err = quad(lambda t: _integrand(t, z, Omega_m, w0, w1), 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quad_dL(nodes, weights, z, Omega_m, w0, w1):
    """Quadrature approximation of dL/(c/H0)"""
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, Omega_m, w0, w1)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    # Define test cosmologies: (Omega_m, w0, w1)
    cosmologies = [
        (0.30, -1.0, 0.0),    # ΛCDM
        (0.27, -1.0, 0.0),    # Planck-like ΛCDM
        (0.30, -1.0, 0.5),    # CPL with w1>0
        (0.30, -1.0, -0.5),   # CPL with w1<0
        (0.30, -0.9, 0.3),    # Another CPL
        (0.30, -1.1, 0.2),    # Phantom-ish
        (0.25, -1.0, 0.0),    # Lower Omega_m
        (0.35, -1.0, 0.0),    # Higher Omega_m
        (0.30, -0.8, -0.3),   # w0 > -1
        (0.30, -1.2, 0.4),    # More phantom
    ]
    
    # Redshifts to test
    redshifts = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Compute exact values
    exact_vals = {}
    for cosmo in cosmologies:
        Om, w0, w1 = cosmo
        for z in redshifts:
            exact_vals[(Om, w0, w1, z)] = _exact_dL(z, Om, w0, w1)
    
    def _objective(params):
        """Worst-case relative error in percent."""
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        # Transform to ensure nodes in (0,1) and increasing, weights positive
        # Use sigmoid-like transform for nodes
        t = 1.0 / (1.0 + np.exp(-nodes_raw))  # sigmoid maps to (0,1)
        # Sort to ensure increasing
        t_sorted = np.sort(t)
        # Ensure strictly in (0,1) with small margin
        t_sorted = np.clip(t_sorted, 1e-6, 1.0 - 1e-6)
        
        # Weights: use softplus to ensure positive
        w = np.exp(weights_raw)
        w = w / np.sum(w)  # normalize to sum to 1
        
        # Compute worst-case relative error
        max_err = 0.0
        for cosmo in cosmologies:
            Om, w0, w1 = cosmo
            for z in redshifts:
                exact = exact_vals[(Om, w0, w1, z)]
                if exact < 1e-10:
                    continue
                approx = _quad_dL(t_sorted, w, z, Om, w0, w1)
                rel_err = abs(approx - exact) / exact
                max_err = max(max_err, rel_err)
        
        return max_err * 100.0  # in percent
    
    # Initial guess: Chebyshev nodes and weights
    n = 8
    # Chebyshev-Gauss nodes
    k = np.arange(1, n + 1)
    nodes_init = 0.5 * (1.0 - np.cos((2 * k - 1) * np.pi / (2 * n)))
    # Invert sigmoid to get initial params
    nodes_params_init = np.log(nodes_init / (1.0 - nodes_init))
    # Equal weights
    weights_params_init = np.zeros(n)
    
    x0 = np.concatenate([nodes_params_init, weights_params_init])
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try a few more local optimizations from perturbed starts
        for trial in range(3):
            if time.time() > deadline - 5.0:
                break
            x_trial = best_x + 0.1 * np.random.randn(16)
            try:
                result = minimize(_objective, x_trial, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x
            except Exception:
                pass
    
    # Extract final nodes and weights
    nodes_raw = best_x[:8]
    weights_raw = best_x[8:16]
    
    t = 1.0 / (1.0 + np.exp(-nodes_raw))
    t_sorted = np.sort(t)
    t_sorted = np.clip(t_sorted, 1e-8, 1.0 - 1e-8)
    
    w = np.exp(weights_raw)
    w = w / np.sum(w)
    
    # Final check: ensure nodes strictly increasing in [0,1], weights positive and sum to 1
    if not np.all(np.diff(t_sorted) > 0):
        # Fallback to midpoint rule
        t_sorted = (np.arange(n) + 0.5) / n
        w = np.full(n, 1.0 / n)
    
    if not np.all(w > 0) or abs(np.sum(w) - 1.0) > 1e-10:
        w = np.full(n, 1.0 / n)
    
    return {"nodes": t_sorted, "weights": w}
