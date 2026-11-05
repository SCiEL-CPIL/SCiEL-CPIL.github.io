"""Combined spectral quadrature design for luminosity distance.

Uses robust parameterization (nodes via tanh, weights via softmax) with a
diverse training set, differential evolution for global search, and
L-BFGS-B / Nelder-Mead local refinement.

The integral is I = ∫_0^1 1/E(a) dz where a = 1/(1+z*node), and the
luminosity distance is (1+z)*z*I. Since (1+z)*z is the same factor for
quadrature and reference, minimizing relative error in I is equivalent.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre


def _E_of_a(a, Om, w0, wa):
    """CPL Hubble expansion E(a)=H(a)/H0 for flat universe."""
    a = np.asarray(a, dtype=float)
    matter = Om * a ** (-3.0)
    if abs(w0 + 1.0) < 1e-10 and abs(wa) < 1e-10:
        de = (1.0 - Om)
    elif abs(w0 + 1.0) < 1e-10:
        exp_arg = 3.0 * wa * (1.0 - a)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.exp(exp_arg)
    else:
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_arg = 3.0 * wa * (1.0 - a) / (1.0 + w0)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.power(a, exponent) * np.exp(exp_arg)
    return np.sqrt(matter + de)


def _integrand_for_nodes(nodes, Om, w0, wa):
    """Compute 1/E(a(z)) at given nodes where a=1/(1+z*node)."""
    a_vals = 1.0 / (1.0 + nodes)
    return 1.0 / _E_of_a(a_vals, Om, w0, wa)


def _ref_integral(z_max, Om, w0, wa, n_gl=128):
    """High-accuracy reference integral via Gauss-Legendre."""
    gz, gw = roots_legendre(n_gl)
    z_pts = 0.5 * z_max * (gz + 1.0)
    w_pts = 0.5 * z_max * gw
    a_pts = 1.0 / (1.0 + z_pts)
    e_vals = _E_of_a(a_pts, Om, w0, wa)
    return np.sum(w_pts / e_vals)


def _gauss_legendre_nodes(n):
    """Gauss-Legendre nodes mapped to [0,1]."""
    gl_nodes, gl_weights = roots_legendre(n)
    return 0.5 * (gl_nodes + 1.0), 0.5 * gl_weights


def _tanh_to_unit(x):
    """Map real parameter to (0,1) via tanh."""
    return 0.5 * (1.0 + np.tanh(x))


def _unit_to_tanh(u):
    """Map (0,1) to real parameter via inverse tanh."""
    return np.arctanh(2.0 * u - 1.0)


def _decode_params(params, n_nodes=8):
    """Decode raw parameters to nodes in (0,1) and positive weights summing to 1."""
    node_params = params[:n_nodes]
    weight_params = params[n_nodes:]
    
    # Nodes via tanh: always in (0,1), strictly increasing if sorted
    nodes = _tanh_to_unit(node_params)
    nodes = np.sort(nodes)
    
    # Weights via softmax: always positive, sum to 1
    w_shifted = weight_params - np.max(weight_params)
    w_exp = np.exp(w_shifted)
    weights = w_exp / np.sum(w_exp)
    
    # Check strict increase
    if np.any(np.diff(nodes) < 1e-12):
        return None, None
    
    return nodes, weights


def _objective(params, models, z_list, ref, n_nodes=8):
    """Objective: worst-case relative error over training set."""
    if time.time() > _DEADLINE:
        return 1e10
    
    nodes, weights = _decode_params(params, n_nodes)
    if nodes is None:
        return 1e8
    
    max_err = 0.0
    for i, (Om, w0, wa) in enumerate(models):
        f_vals = _integrand_for_nodes(nodes, Om, w0, wa)
        q = np.dot(weights, f_vals)
        for j in range(len(z_list)):
            rv = ref[i, j]
            if abs(rv) < 1e-15:
                continue
            rel_err = abs(q - rv) / abs(rv)
            if rel_err > max_err:
                max_err = rel_err
    return max_err


_DEADLINE = 0.0


def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 82.0  # Safety margin under 90s kill

    # --- Training cosmologies (focused set for fast evaluation) ---
    models = []
    Om_list = [0.2, 0.3, 0.4, 0.5]
    w0_list = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_list = [-0.5, -0.1, 0.0, 0.3, 0.5]
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                models.append((Om, w0, wa))

    z_list = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Pre-compute reference integrals
    n_models = len(models)
    n_z = len(z_list)
    ref = np.zeros((n_models, n_z))
    for i, (Om, w0, wa) in enumerate(models):
        for j, zm in enumerate(z_list):
            ref[i, j] = _ref_integral(zm, Om, w0, wa)

    n_nodes = 8

    # --- Initialization: Gauss-Legendre (strong prior) ---
    best_val = np.inf
    best_x = None

    gl_nodes, gl_weights = _gauss_legendre_nodes(n_nodes)
    # Convert to tanh parameterization
    u_init = _unit_to_tanh(gl_nodes)
    v_init = np.log(gl_weights + 1e-15)
    x_init = np.concatenate([u_init, v_init])
    val = _objective(x_init, models, z_list, ref, n_nodes)
    if val < best_val:
        best_val = val
        best_x = x_init.copy()

    # --- Phase 1: Differential Evolution ---
    if time.time() < _DEADLINE - 30.0:
        bounds = [(-8.0, 8.0)] * n_nodes + [(-15.0, 15.0)] * n_nodes
        try:
            de_res = differential_evolution(
                _objective, bounds, args=(models, z_list, ref, n_nodes),
                maxiter=100, popsize=12, tol=1e-10, seed=12345,
                polish=False, init='latinhypercube',
                callback=lambda xk, conv: (time.time() > _DEADLINE - 30.0)
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

    # --- Phase 2: L-BFGS-B refinement ---
    if time.time() < _DEADLINE - 15.0:
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B',
                           args=(models, z_list, ref, n_nodes),
                           bounds=[(-20.0, 20.0)] * n_nodes + [(-30.0, 30.0)] * n_nodes,
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Phase 3: Nelder-Mead final polish ---
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           args=(models, z_list, ref, n_nodes),
                           options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Decode and finalize ---
    nodes, weights = _decode_params(best_x, n_nodes)
    if nodes is None:
        # Fallback: return GL nodes
        gl_nodes, gl_weights = _gauss_legendre_nodes(n_nodes)
        nodes = gl_nodes
        weights = gl_weights / np.sum(gl_weights)
    
    # Enforce strict increase
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12

    # Enforce positivity and normalize
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
