"""Rational-function (Padé-like) quadrature design for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL(z, w0, wa):
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * val

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _worst_rel_error(nodes, weights, test_cosmos, test_zs):
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _gauss_legendre_init():
    """Get 8-point Gauss-Legendre nodes and weights on [0,1]."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _solve_pade_quadrature(n_nodes=8, n_pade_num=4, n_pade_den=3):
    """
    Solve for quadrature nodes and weights using a rational (Padé-like) approximation.
    
    Idea: We want to find nodes x_i and weights w_i such that for a representative
    set of cosmologies, the quadrature sum approximates the true integral well.
    
    Instead of directly optimizing nodes, we use a two-stage approach:
    1. Choose a reference set of cosmologies
    2. For a given set of nodes, solve for optimal weights via linear least squares
    3. Optimize nodes to minimize worst-case error
    
    The rational function aspect: we approximate 1/E(a) as a rational function
    R(a) = P(a)/Q(a) where P is degree n_pade_num and Q is degree n_pade_den.
    Then the quadrature becomes exact for this rational family, and we choose
    nodes to minimize error against the true 1/E(a).
    """
    # Representative cosmologies for training
    train_cosmos = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Lambda
        (0.5, -1.0),     # Quintessence
        (-1.5, 0.5),     # Phantom
        (-0.5, 0.5),     # Another CPL
        (0.3, -0.3),     # Mild CPL
        (0.0, 0.5),      # Positive wa
        (-1.0, 1.0),     # Strong phantom
        (1.0, -1.0),     # Strong quintessence
        (-0.3, -0.3),    # Mild negative
        (0.7, -0.7),     # More quintessence
        (-0.8, 0.3),     # More phantom
    ]
    train_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    # Stage 1: For a given set of nodes, solve for optimal weights
    def _solve_weights(nodes):
        """Solve for weights that minimize worst-case relative error."""
        # Build constraint matrix: for each (cosmo, z), we have
        # sum_i w_i / E(a_i) = true_integral / (z)
        # where a_i = 1/(1 + z * nodes_i)
        
        n_constraints = len(train_cosmos) * len(train_zs)
        A = np.zeros((n_constraints, n_nodes))
        b = np.zeros(n_constraints)
        
        idx = 0
        for w0, wa in train_cosmos:
            for z in train_zs:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                integrand_vals = 1.0 / E_vals
                
                true_val = _true_dL(z, w0, wa)
                # integral_approx = sum(w_i * integrand_vals_i)
                # true integral = true_val / (1+z) ... wait
                # dL/(c/H0) = (1+z) * integral_0^z dz'/E(a')
                # So integral_0^z = true_val / (1+z)
                target = true_val / (1.0 + z)
                
                A[idx, :] = integrand_vals
                b[idx] = target
                idx += 1
        
        # Solve least squares with non-negativity constraint
        # Use scipy's nnls or project after
        from scipy.optimize import nnls
        w_sol, rnorm = nnls(A, b)
        
        # Normalize to sum to 1
        w_sum = np.sum(w_sol)
        if w_sum > 1e-15:
            w_sol = w_sol / w_sum
        else:
            w_sol = np.ones(n_nodes) / n_nodes
        
        return w_sol
    
    # Stage 2: Optimize nodes
    def _node_objective(params):
        if _time_left() < 2.0:
            return 1e10
        
        # Decode nodes from params using sigmoid
        nodes = 1.0 / (1.0 + np.exp(-params))
        nodes = np.sort(nodes)
        for i in range(1, n_nodes):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Solve for optimal weights
        weights = _solve_weights(nodes)
        
        # Evaluate worst-case error
        return _worst_rel_error(nodes, weights, train_cosmos, train_zs)
    
    # Initial guess: Gauss-Legendre nodes
    gl_nodes, gl_weights = _gauss_legendre_init()
    params_init = np.log(gl_nodes / (1.0 - gl_nodes))
    
    best_params = params_init.copy()
    best_score = _node_objective(best_params)
    
    # Optimize nodes with Nelder-Mead
    if _time_left() > 10.0:
        try:
            res = minimize(_node_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Decode final nodes and solve for final weights
    nodes = 1.0 / (1.0 + np.exp(-best_params))
    nodes = np.sort(nodes)
    for i in range(1, n_nodes):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    weights = _solve_weights(nodes)
    
    # Ensure weights are positive and sum to 1
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return nodes, weights

def _refine_with_validation(nodes, weights):
    """Refine weights using a broader validation set."""
    val_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (0.0, 0.5),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-0.3, -0.3),
        (0.7, -0.7),
        (-0.8, 0.3),
        (-0.2, -0.5),
        (0.2, 0.2),
    ]
    val_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
    
    n_nodes = len(nodes)
    n_constraints = len(val_cosmos) * len(val_zs)
    A = np.zeros((n_constraints, n_nodes))
    b = np.zeros(n_constraints)
    
    idx = 0
    for w0, wa in val_cosmos:
        for z in val_zs:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            integrand_vals = 1.0 / E_vals
            true_val = _true_dL(z, w0, wa)
            target = true_val / (1.0 + z)
            A[idx, :] = integrand_vals
            b[idx] = target
            idx += 1
    
    from scipy.optimize import nnls
    w_sol, rnorm = nnls(A, b)
    w_sum = np.sum(w_sol)
    if w_sum > 1e-15:
        w_sol = w_sol / w_sum
    else:
        w_sol = np.ones(n_nodes) / n_nodes
    
    return nodes, w_sol

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Primary approach: rational-function-inspired quadrature design
    try:
        nodes, weights = _solve_pade_quadrature(n_nodes=8, n_pade_num=4, n_pade_den=3)
    except Exception:
        # Fallback to Gauss-Legendre
        nodes, weights = _gauss_legendre_init()
    
    # Refine weights with broader validation
    if _time_left() > 10.0:
        try:
            nodes, weights = _refine_with_validation(nodes, weights)
        except Exception:
            pass
    
    # Ensure validity
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
