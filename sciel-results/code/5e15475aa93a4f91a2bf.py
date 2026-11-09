"""Hierarchical 8-node quadrature: LP for weights, Nelder-Mead for nodes."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

_DEADLINE = None

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _E_cpl(a, w0, wa):
    """H(a)/H0 for flat CPL model (no radiation, standard cosmology)."""
    a = np.asarray(a, dtype=float)
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10)
    denom = 1.0 + w0
    if abs(denom) < 1e-10:
        exponent = 0.0
    else:
        exponent = wa / denom
    return a**(-3.0) * base**exponent

def _integrand_at_z(z, w0, wa):
    """1/E(z) for CPL model."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)

def _exact_integral(z, w0, wa):
    """Compute integral_0^z dz'/E(z') using adaptive quadrature."""
    result, _ = quad(lambda zp: _integrand_at_z(zp, w0, wa), 0, z, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result

def _quad_integral(z, nodes, weights, w0, wa):
    """Quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    vals = 1.0 / E_vals
    return z * np.dot(weights, vals)

def _solve_weights_lp(nodes, exact_integrals, model_params, z_vals):
    """
    For fixed nodes, solve:
      min t
      s.t. for all (m, z): |z * sum_i w_i / E(a_i; m, z) - exact_mz| <= t * |exact_mz|
           sum_i w_i = 1
           w_i > 0
    This is a linear program in variables (w_1..w_8, t).
    """
    n_models = len(model_params)
    n_z = len(z_vals)
    n_constraints = n_models * n_z
    
    # Pre-compute integrand values at nodes for each (model, z)
    # F[m, z, i] = z * 1/E(a_i; model, z)
    F = np.zeros((n_models, n_z, 8))
    for mi, (w0, wa) in enumerate(model_params):
        for zi, z in enumerate(z_vals):
            z_nodes = z * nodes
            a_nodes = 1.0 / (1.0 + z_nodes)
            E_vals = _E_cpl(a_nodes, w0, wa)
            F[mi, zi, :] = z / E_vals
    
    # LP variables: w_1..w_8, t  (total 9 variables)
    # Minimize: t  -> c = [0,0,0,0,0,0,0,0, 1]
    c = np.zeros(9)
    c[8] = 1.0
    
    # Constraints: for each (m, z):
    #   F[m,z,:] @ w - t*|exact| <= 0
    #  -F[m,z,:] @ w + t*|exact| <= 0
    A_ub = []
    b_ub = []
    idx = 0
    for mi in range(n_models):
        for zi in range(n_z):
            exact = exact_integrals[idx]
            idx += 1
            if abs(exact) < 1e-15:
                continue
            abs_exact = abs(exact)
            # Row 1: F @ w - t*abs_exact <= 0
            row1 = np.zeros(9)
            row1[:8] = F[mi, zi, :]
            row1[8] = -abs_exact
            A_ub.append(row1)
            b_ub.append(0.0)
            # Row 2: -F @ w + t*abs_exact <= 0
            row2 = np.zeros(9)
            row2[:8] = -F[mi, zi, :]
            row2[8] = abs_exact
            A_ub.append(row2)
            b_ub.append(0.0)
    
    if not A_ub:
        # Degenerate: just use equal weights
        return np.ones(8) / 8.0, 0.0
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, 9))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 1e-12 (positive), t >= 0
    bounds = [(1e-12, None)] * 8 + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            weights = res.x[:8]
            t_val = res.x[8]
            # Normalize to be safe
            weights = weights / np.sum(weights)
            return weights, t_val
        else:
            # Fallback: Gauss-Legendre weights
            _, gl_w = np.polynomial.legendre.leggauss(8)
            gl_w = 0.5 * gl_w
            return gl_w / np.sum(gl_w), 1e10
    except Exception:
        _, gl_w = np.polynomial.legendre.leggauss(8)
        gl_w = 0.5 * gl_w
        return gl_w / np.sum(gl_w), 1e10

def _node_objective(nodes, exact_integrals, model_params, z_vals):
    """
    Objective for node optimization: solve LP for weights, return worst relative error.
    """
    if _time_left() < 3.0:
        return 1e10
    
    # Ensure nodes are valid: strictly increasing in (0, 1)
    if not np.all(np.diff(nodes) > 1e-12):
        return 1e10
    if nodes[0] < 1e-10 or nodes[-1] > 1.0 - 1e-10:
        return 1e10
    
    weights, t_val = _solve_weights_lp(nodes, exact_integrals, model_params, z_vals)
    
    if t_val > 1e10:
        return 1e10
    
    return t_val

def _params_to_nodes(params):
    """
    Convert 8 parameters to 8 strictly increasing nodes in (0, 1).
    params[i] represents log-spacing between consecutive nodes.
    nodes[0] = exp(params[0]) / sum(exp(params[:1]))
    Actually, let's use a simpler encoding:
    params[0] = log(node_0)
    params[i] = log(node_i - node_{i-1}) for i=1..7
    Then normalize so nodes are in (0,1).
    """
    # node_0 = exp(params[0])
    # node_i = node_{i-1} + exp(params[i]) for i=1..7
    node_0 = np.exp(params[0])
    spacings = np.exp(params[1:8])
    raw_nodes = np.zeros(8)
    raw_nodes[0] = node_0
    for i in range(1, 8):
        raw_nodes[i] = raw_nodes[i-1] + spacings[i-1]
    
    # Normalize to (0, 1): scale so max is 1
    scale = raw_nodes[-1]
    if scale < 1e-15:
        return np.linspace(0.01, 0.99, 8)
    nodes = raw_nodes / scale
    
    # Ensure strictly increasing and in (0,1)
    nodes = np.maximum(nodes, 1e-10)
    nodes = np.minimum(nodes, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-12
    
    return nodes

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 85.0  # 5s safety margin
    
    # Representative CPL models (w0, wa)
    models = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Lambda
        (-0.5, -0.5),    # Quintessence-like
        (-1.0, 0.5),     # Phantom-like
        (-0.8, -0.5),    # Another CPL
        (-1.2, 0.3),     # Strong phantom
        (-0.3, -0.3),    # Mild CPL
        (-1.0, -0.5),    # Another
        (0.5, -1.0),     # w0 > 0 case
        (-1.5, 0.5),     # Extreme phantom
    ]
    
    # Test redshifts
    z_test = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Pre-compute exact integrals
    exact_integrals = []
    for w0, wa in models:
        for z in z_test:
            try:
                val = _exact_integral(z, w0, wa)
                exact_integrals.append(val)
            except Exception:
                exact_integrals.append(1.0)
    exact_integrals = np.array(exact_integrals)
    
    model_params = models
    z_vals = z_test
    
    # Initial guess: Gauss-Legendre nodes
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    
    # Encode initial nodes as params
    def nodes_to_params(nodes):
        params = np.zeros(8)
        params[0] = np.log(nodes[0])
        for i in range(1, 8):
            params[i] = np.log(nodes[i] - nodes[i-1])
        return params
    
    params0 = nodes_to_params(nodes0)
    
    # First: solve LP for weights with GL nodes
    weights0, err0 = _solve_weights_lp(nodes0, exact_integrals, model_params, z_vals)
    best_params = params0.copy()
    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    best_err = err0
    
    # Optimize nodes using Nelder-Mead
    bounds = [(-2.0, 2.0)] * 8  # Reasonable range for log-spacings
    
    def objective(params):
        if _time_left() < 5.0:
            return 1e10
        nodes = _params_to_nodes(params)
        return _node_objective(nodes, exact_integrals, model_params, z_vals)
    
    # Try multiple restarts with different initial points
    np.random.seed(42)
    restarts = [
        params0,  # GL nodes
        params0 + np.random.randn(8) * 0.1,
        params0 + np.random.randn(8) * 0.2,
    ]
    
    for restart_idx, p_init in enumerate(restarts):
        if _time_left() < 20.0:
            break
        
        try:
            res = minimize(
                objective,
                p_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                best_nodes = _params_to_nodes(res.x)
                # Re-solve LP for final weights
                best_weights, _ = _solve_weights_lp(best_nodes, exact_integrals, model_params, z_vals)
                
        except Exception:
            pass
    
    # Final refinement with tighter tolerances
    if _time_left() > 15.0:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                best_nodes = _params_to_nodes(res.x)
                best_weights, _ = _solve_weights_lp(best_nodes, exact_integrals, model_params, z_vals)
        except Exception:
            pass
    
    # Final validation and cleanup
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    # Ensure strictly increasing
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Verify weights sum to 1 within tolerance
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
