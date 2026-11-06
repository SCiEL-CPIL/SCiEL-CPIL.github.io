"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat CPL model. a = 1/(1+z)."""
    # E^2(a) = Om_m * a^-3 + Om_k * a^-2 + Om_de * a^(-3(1+w0+wa*(1-a)))
    Om_m = 0.3
    Om_de = 0.7
    Om_k = 0.0
    # General: E^2 = Om_m a^-3 + Om_de a^(-3(1+w0+wa*(1-a)))
    # For CPL: w(a) = w0 + wa*(1-a)
    # E^2 = Om_m a^-3 + Om_de a^(-3(1+w0+wa*(1-a)))
    exponent = -3.0 * (1.0 + w0 + wa * (1.0 - a))
    E2 = Om_m * a**(-3.0) + Om_de * a**exponent
    return np.sqrt(E2)

def _integrand_z(z, w0, wa):
    """f(z) = (1+z)*z / E(1/(1+z))"""
    a = 1.0 / (1.0 + z)
    E = _E_of_a(a, w0, wa)
    return (1.0 + z) * z / E

def _true_integral(z_eval, w0, wa, n_quad=2000):
    """Compute true integral using fine trapezoidal rule."""
    # z_eval is array of z values where we want the integral from 0 to z_eval
    # Actually we need the full integral 0 to 1, and also partial?
    # The quadrature computes the full integral 0 to 1.
    # Let's compute the full integral with high accuracy.
    z_fine = np.linspace(0, 1, n_quad)
    f_fine = _integrand_z(z_fine, w0, wa)
    # Handle z=0: f(0) = 0
    from scipy.integrate import quad
    val, _ = quad(lambda z: _integrand_z(z, w0, wa), 0, 1, limit=500)
    return val

def _quadrature_value(nodes, weights, w0, wa):
    """Compute quadrature approximation of the integral."""
    a = 1.0 / (1.0 + nodes)
    E = _E_of_a(a, w0, wa)
    f_vals = (1.0 + nodes) * nodes / E
    return np.sum(weights * f_vals)

def _relative_error(nodes, weights, w0, wa):
    """Relative error of quadrature vs true integral."""
    true_val = _true_integral(None, w0, wa)
    quad_val = _quadrature_value(nodes, weights, w0, wa)
    return abs(quad_val - true_val) / abs(true_val)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Start with Gauss-Legendre on [0,1]
    x_gl, w_gl = roots_legendre(8)
    nodes0 = 0.5 * (x_gl + 1.0)
    weights0 = 0.5 * w_gl
    
    # Define diverse cosmology models (w0, wa)
    models = [
        (0.0, 0.0),      # Lambda CDM
        (-1.0, 0.0),     # Cosmological constant
        (-1.0, 0.5),     # CPL
        (-1.0, -0.5),    # CPL
        (-1.5, 1.0),     # Phantom-ish
        (-0.5, -1.0),    # Quintessence
        (-1.0, 1.0),     # CPL
        (-1.0, -1.0),    # CPL
        (-2.0, 0.0),     # Phantom
        (-0.8, 0.3),     # Mild CPL
        (-1.2, -0.3),    # Mild CPL
        (-1.0, 2.0),     # Strong CPL
        (-1.0, -2.0),    # Strong CPL
        (-0.3, -0.7),    # Quintessence
        (-1.5, -0.5),    # Phantom CPL
        (-0.7, 0.7),     # Mild
    ]
    
    # Precompute true integrals
    true_vals = []
    for w0, wa in models:
        tv = _true_integral(None, w0, wa)
        true_vals.append(tv)
    true_vals = np.array(true_vals)
    
    def _objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce nodes in (0,1) and strictly increasing
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        # Weights positive
        if np.any(weights <= 0):
            return 1e10
        # Weights sum to 1
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
        
        max_err = 0.0
        for i, (w0, wa) in enumerate(models):
            a = 1.0 / (1.0 + nodes)
            E = _E_of_a(a, w0, wa)
            f_vals = (1.0 + nodes) * nodes / E
            quad_val = np.dot(weights, f_vals)
            rel_err = abs(quad_val - true_vals[i]) / abs(true_vals[i])
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    # Optimize
    x0 = np.concatenate([nodes0, weights0])
    best_params = x0.copy()
    best_obj = _objective(x0)
    
    # Try Nelder-Mead
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Try a few more restarts with perturbed Gauss-Legendre
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        pert_nodes = nodes0 + rng.normal(0, 0.05, 8)
        pert_weights = weights0 + rng.normal(0, 0.02, 8)
        # Normalize
        pert_weights = np.abs(pert_weights)
        pert_weights = pert_weights / np.sum(pert_weights)
        # Sort nodes
        order = np.argsort(pert_nodes)
        pert_nodes = pert_nodes[order]
        pert_weights = pert_weights[order]
        # Clip to valid range
        pert_nodes = np.clip(pert_nodes, 0.01, 0.99)
        # Ensure strictly increasing
        for i in range(1, 8):
            if pert_nodes[i] <= pert_nodes[i-1]:
                pert_nodes[i] = pert_nodes[i-1] + 0.01
        if pert_nodes[-1] > 0.99:
            break
        
        x_start = np.concatenate([pert_nodes, pert_weights])
        try:
            result = minimize(_objective, x_start, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    nodes = best_params[:8]
    weights = best_params[8:]
    
    # Final cleanup: ensure constraints
    # Sort by node
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    # Ensure strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    
    # Ensure positive weights summing to 1
    weights = np.abs(weights)
    if np.sum(weights) < 1e-15:
        weights = np.full(8, 0.125)
    else:
        weights = weights / np.sum(weights)
    
    # Verify sum is within 1e-12
    s = np.sum(weights)
    if abs(s - 1.0) > 1e-12:
        weights = weights / s
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
