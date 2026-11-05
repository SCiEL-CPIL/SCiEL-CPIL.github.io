"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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
    return trapezoid(f, t_grid)

def _compute_errors(nodes, weights, portfolio, z_grid):
    """Compute relative errors for given nodes/weights across portfolio and z_grid."""
    errors = []
    for Om, w0, wa in portfolio:
        for z in z_grid:
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = _reference_value_fast(z, Om, w0, wa, n_ref=1500)
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                errors.append(rel_err)
    return np.array(errors) if errors else np.array([1e10])

def _soft_max(errors, beta=50.0):
    """Smooth max via log-sum-exp."""
    errors = np.clip(errors, 0, 1e6)
    max_e = np.max(errors)
    if max_e < 1e-10:
        return max_e
    shifted = errors - max_e
    log_sum_exp = max_e + np.log(np.sum(np.exp(beta * shifted)))
    return (1.0 / beta) * log_sum_exp

def _objective_smooth(x, portfolio, z_grid, beta=50.0):
    """Objective using soft-max smoothing."""
    nodes = x[:8]
    weights = x[8:]
    
    # Check constraints with penalty
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e10
    
    errors = _compute_errors(nodes, weights, portfolio, z_grid)
    return _soft_max(errors, beta)

def _objective_hard(x, portfolio, z_grid):
    """Objective using hard max."""
    nodes = x[:8]
    weights = x[8:]
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e10
    
    errors = _compute_errors(nodes, weights, portfolio, z_grid)
    return np.max(errors)

def _tanh_nodes(raw_nodes):
    """Map raw nodes to (0,1) using tanh-based parameterization."""
    # Map from R^8 to (0,1)^8 with strict ordering
    # Use cumulative sum of positive values
    raw = np.abs(raw_nodes) + 0.01
    cumsum = np.cumsum(raw)
    nodes = cumsum / cumsum[-1]
    # Ensure strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    return nodes

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
        (0.28, -1.05, 0.1),
        (0.29, -0.95, -0.1),
    ]
    
    z_grid = np.linspace(0.1, 2.0, 40)
    
    # Start from Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes0 = (nodes_gl + 1.0) / 2.0
    weights0 = weights_gl / 2.0
    x0 = np.concatenate([nodes0, weights0])
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() - t_start < deadline - 40.0:
        bounds_de = [(-1.0, 1.0)] * 8 + [(0.0, 1.0)] * 8
        
        def _de_objective(x_de):
            if time.time() - t_start > deadline - 40.0:
                return 1e10
            raw_nodes = x_de[:8]
            weights_raw = x_de[8:]
            
            # Transform nodes using tanh-based parameterization
            nodes = _tanh_nodes(raw_nodes)
            
            # Normalize weights
            weights = np.maximum(weights_raw, 1e-10)
            weights = weights / np.sum(weights)
            
            x = np.concatenate([nodes, weights])
            return _objective_smooth(x, portfolio, z_grid, beta=30.0)
        
        try:
            de_result = differential_evolution(
                _de_objective, bounds_de,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.9
            )
            if de_result.fun < _objective_smooth(x0, portfolio, z_grid, beta=30.0):
                x0 = np.concatenate([_tanh_nodes(de_result.x[:8]),
                                     np.maximum(de_result.x[8:], 1e-10)])
                x0[8:] = x0[8:] / np.sum(x0[8:])
        except Exception:
            pass
    
    # Phase 2: SLSQP local refinement with smooth objective
    best_x = x0.copy()
    best_obj = _objective_smooth(best_x, portfolio, z_grid, beta=50.0)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
    ]
    bounds = [(1e-6, 1 - 1e-6)] * 8 + [(1e-8, 1.0)] * 8
    
    n_iter = 0
    while time.time() - t_start < deadline - 15.0:
        n_iter += 1
        try:
            res = minimize(_objective_smooth, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, args=(portfolio, z_grid, 50.0),
                         options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            break
    
    # Phase 3: Hard max polishing
    if time.time() - t_start < deadline - 5.0:
        try:
            res = minimize(_objective_hard, best_x, method='SLSQP', bounds=bounds,
                         constraints=constraints, args=(portfolio, z_grid),
                         options={'maxiter': 300, 'ftol': 1e-14})
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
