"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _reference_integrand(z, Om, w0, wa):
    """Compute z/(1+z) * 1/E(a) for CPL model."""
    if z < 1e-10:
        return 0.0
    a = 1.0 / (1.0 + z)
    # E(a) = (Om/a^3 + Ode * a^(-3w0) * exp(-3*wa*(1-a)))^0.5
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a))
    E2 = term1 + term2
    if E2 <= 0:
        E2 = 1e-30
    E = np.sqrt(E2)
    return z / (1.0 + z) / E

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a_vals**3)
    term2 = Ode * (a_vals**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a_vals))
    E2 = term1 + term2
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.sum(weights / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=2000):
    """High-accuracy reference using fine quadrature with vectorization."""
    z_grid = np.linspace(0, z, n_ref)
    a_vals = 1.0 / (1.0 + z_grid)
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a_vals**3)
    term2 = Ode * (a_vals**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a_vals))
    E2 = term1 + term2
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    f_vals = z_grid / (1.0 + z_grid) / E
    f_vals[0] = 0.0  # handle z=0 singularity
    from scipy.integrate import trapezoid
    return trapezoid(f_vals, z_grid)

def _worst_case_error(nodes, weights, portfolio, z_max=2.0, n_z=40):
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
    # Use soft-max to smooth the max operation
    beta = 50.0
    log_sum_exp = np.log(np.sum(np.exp(beta * errors)))
    return (1.0 / beta) * log_sum_exp

def construct():
    t_start = time.time()
    deadline = 85.0  # seconds, leave 5s margin
    
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
    x0 = np.concatenate([(nodes_gl + 1.0)/2.0, weights_gl/2.0])
    
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
    bounds = [(1e-6, 1-1e-6)]*8 + [(1e-8, 1.0)]*8
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization with soft-max
    n_iter = 0
    while time.time() - t_start < deadline - 10.0:
        n_iter += 1
        try:
            res = minimize(_objective, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 150, 'ftol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            break
    
    # Phase 2: Fine polishing with tighter tolerance
    if time.time() - t_start < deadline - 2.0:
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
    nodes = np.clip(nodes, 1e-8, 1-1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
