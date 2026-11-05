"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapezoid
import time

def _E_squared(a, Om, w0, wa):
    """Vectorized E^2(a) for CPL model."""
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0 * w0)) * np.exp(-3.0 * wa * (1.0 - a))
    return np.maximum(term1 + term2, 1e-30)

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.dot(weights, 1.0 / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=2000):
    """High-accuracy reference using vectorized trapezoidal rule on t in [0,1]."""
    if z < 1e-12:
        return 0.0
    t_grid = np.linspace(0, 1, n_ref)
    a_vals = 1.0 / (1.0 + z * t_grid)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    f = 1.0 / E
    # Integrand is 1/E(a(zt)) dt, and dL/(c/H0) = (1+z)*z * integral
    # But we need to be careful: the integral is ∫0^1 1/E(a(zt)) dt
    # Actually looking at the formula: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So the integral being approximated is ∫0^1 1/E(a(zt)) dt
    # The reference should compute this integral accurately
    integral_val = trapezoid(f, t_grid)
    return (1.0 + z) * z * integral_val

def _worst_case_error(nodes, weights, portfolio, z_max=2.0, n_z=50):
    """Compute worst relative error across portfolio with soft-max smoothing."""
    errors = []
    z_grid = np.linspace(0.1, z_max, n_z)
    for Om, w0, wa in portfolio:
        for z in z_grid:
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = _reference_value_fast(z, Om, w0, wa, n_ref=1500)
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                errors.append(rel_err)
    if not errors:
        return 1e10
    errors = np.array(errors)
    # Use soft-max to smooth the max operation for better gradient flow
    beta = 50.0
    max_err = np.max(errors)
    # Clip for numerical stability
    clipped = np.clip(errors - max_err, -50.0, 50.0)
    log_sum_exp = np.log(np.sum(np.exp(beta * clipped)))
    return max_err + (1.0 / beta) * log_sum_exp

def construct():
    t_start = time.time()
    deadline = 85.0  # seconds, leave 5s margin for safety
    
    # Portfolio of cosmologies to optimize for
    portfolio = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.3),   # CPL with positive wa
        (0.30, -1.0, -0.3),  # CPL with negative wa
        (0.27, -1.0, 0.0),
        (0.27, -1.2, 0.5),
        (0.27, -0.8, -0.5),
        (0.31, -1.1, 0.2),
        (0.31, -0.9, -0.2),
    ]
    
    # Start from Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    x0 = np.concatenate([(nodes_gl + 1.0) / 2.0, weights_gl / 2.0])
    
    def _objective(x):
        if time.time() - t_start > deadline:
            return 1e10
        nodes = x[:8]
        weights = x[8:]
        
        # Check constraints
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
        
        return _worst_case_error(nodes, weights, portfolio)
    
    # Use SLSQP for constrained optimization
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
    ]
    bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-8, 1.0)] * 8
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization with soft-max
    n_iter = 0
    while time.time() - t_start < deadline - 15.0:
        n_iter += 1
        try:
            res = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 150, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            break
        
        if time.time() - t_start > deadline - 15.0:
            break
    
    # Phase 2: Fine polishing with tighter tolerance
    if time.time() - t_start < deadline - 3.0:
        try:
            res = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
    
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Ensure constraints are met
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
