"""
Optimized 8-node positive quadrature rule for dimensionless luminosity distance.
Uses differential evolution for node placement, then solves for optimal weights.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """Compute E(a) = H(a)/H0 for CPL dark energy model."""
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

# Precompute high-accuracy reference quadrature
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(96)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    """Compute true dL/(c/H0) using high-order Gauss-Legendre."""
    if z < 1e-10:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _compute_F_T(nodes, test_cosmos, test_zs):
    """Compute design matrix F and target vector T."""
    num_tests = len(test_cosmos) * len(test_zs)
    F = np.zeros((num_tests, 8))
    T = np.zeros(num_tests)
    
    idx = 0
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z < 1e-10:
                T[idx] = 1.0
                F[idx, :] = 1.0
            else:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                F[idx, :] = 1.0 / E_vals
                true_val = _true_dL(z, w0, wa)
                T[idx] = true_val / ((1.0 + z) * z)
            idx += 1
    return F, T

def _solve_weights_lp(F, T):
    """Solve LP to minimize worst-case relative error."""
    n_vars = 8
    n_tests = len(T)
    c = np.zeros(n_vars + 1)
    c[-1] = 1.0
    A_ub = np.zeros((2 * n_tests, n_vars + 1))
    b_ub = np.zeros(2 * n_tests)
    for j in range(n_tests):
        t_j = T[j]
        if t_j < 1e-10:
            continue
        A_ub[2*j, :n_vars] = F[j, :]
        A_ub[2*j, -1] = -t_j
        b_ub[2*j] = t_j
        A_ub[2*j+1, :n_vars] = -F[j, :]
        A_ub[2*j+1, -1] = t_j
        b_ub[2*j+1] = t_j
    A_eq = np.zeros((1, n_vars + 1))
    A_eq[0, :n_vars] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n_vars + [(0, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[:n_vars]
    except Exception:
        pass
    return None

def _solve_weights_ls(F, T):
    """Solve weighted least squares for weights."""
    n_vars = 8
    W = 1.0 / T
    W[~np.isfinite(W)] = 0
    W[W < 1e10] = 0
    
    Fw = F * W[:, None]
    Tw = T * W
    
    def obj(w):
        r = Fw @ w - Tw
        return np.dot(r, r)
    
    def grad(w):
        r = Fw @ w - Tw
        return 2.0 * Fw.T @ r
    
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(8)}]
    bounds = [(0, None)] * 8
    w0 = np.ones(8) / 8.0
    
    try:
        res = minimize(obj, w0, method='SLSQP', jac=grad, bounds=bounds, constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-14})
        if res.success and res.x is not None:
            w = np.maximum(res.x, 0)
            s = np.sum(w)
            if s > 1e-15:
                return w / s
    except Exception:
        pass
    return None

def _worst_rel_error(nodes, test_cosmos, test_zs):
    """Compute worst-case relative error for given nodes."""
    F, T = _compute_F_T(nodes, test_cosmos, test_zs)
    
    weights = _solve_weights_lp(F, T)
    if weights is None:
        weights = _solve_weights_ls(F, T)
    if weights is None:
        return 1e10, np.ones(8) / 8.0
    
    approx = F @ weights
    rel_errors = np.abs(approx - T) / T
    rel_errors = rel_errors[np.isfinite(rel_errors)]
    worst = np.max(rel_errors) if len(rel_errors) > 0 else 1e10
    return worst, weights

def _decode_params(params):
    """Decode sigmoid parameters to sorted nodes in (0,1)."""
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    return nodes

def _encode_initial(nodes):
    """Encode nodes to sigmoid parameter space."""
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    return -np.log(1.0/nodes - 1.0)

def _gauss_legendre_init():
    """Get Gauss-Legendre nodes as initial guess."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    idx = np.argsort(nodes)
    return nodes[idx]

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Test cosmologies and redshifts
    test_cosmos = [
        (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
        (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
        (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
        (0.2, -0.5), (-0.8, 0.2), (1.5, -1.5), (-2.5, 2.0)
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    # Initial guess from Gauss-Legendre
    gl_nodes = _gauss_legendre_init()
    best_nodes = gl_nodes.copy()
    best_err, best_weights = _worst_rel_error(best_nodes, test_cosmos, test_zs)
    
    # Differential evolution for node optimization
    if _time_left() > 20.0:
        bounds_de = [(-10, 10)] * 8  # Sigmoid parameter bounds
        
        def _de_objective(params):
            if _time_left() < 5.0:
                return 1e10
            nodes = _decode_params(params)
            # Check strict increasing
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1] + 1e-8:
                    return 1e10
            err, _ = _worst_rel_error(nodes, test_cosmos, test_zs)
            return err
        
        try:
            np.random.seed(42)
            de_res = differential_evolution(
                _de_objective, bounds_de,
                maxiter=50, popsize=15, tol=1e-8,
                mutation=(0.3, 1.0), recombination=0.8,
                seed=42, polish=True, init='sobol'
            )
            if de_res.fun < best_err:
                best_err = de_res.fun
                best_nodes = _decode_params(de_res.x)
        except Exception:
            pass
    
    # Nelder-Mead polish
    if _time_left() > 10.0:
        def _nm_objective(params):
            if _time_left() < 3.0:
                return 1e10
            nodes = _decode_params(params)
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1] + 1e-8:
                    return 1e10
            err, _ = _worst_rel_error(nodes, test_cosmos, test_zs)
            return err
        
        init_params = _encode_initial(best_nodes)
        try:
            nm_res = minimize(_nm_objective, init_params, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if nm_res.fun < best_err:
                best_err = nm_res.fun
                best_nodes = _decode_params(nm_res.x)
        except Exception:
            pass
    
    # Final weight optimization
    F_final, T_final = _compute_F_T(best_nodes, test_cosmos, test_zs)
    weights = _solve_weights_lp(F_final, T_final)
    if weights is None:
        weights = _solve_weights_ls(F_final, T_final)
    if weights is None:
        weights = np.ones(8) / 8.0
    
    weights = np.abs(weights)
    s = np.sum(weights)
    if s > 1e-15:
        weights = weights / s
    else:
        weights = np.ones(8) / 8.0
    
    # Ensure nodes are strictly increasing and in (0,1)
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-6
    
    return {"nodes": best_nodes, "weights": weights}
