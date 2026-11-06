"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    t_start = time.time()
    deadline = t_start + 82.0

    # Design cosmologies: broader coverage in parameter space
    design_cosmos = [
        (0.30, 0.70, -1.00,  0.00),
        (0.27, 0.73, -1.00,  0.00),
        (0.30, 0.70, -1.00,  1.00),
        (0.30, 0.70, -1.00, -1.00),
        (0.20, 0.80, -1.20,  0.50),
        (0.35, 0.65, -0.80,  0.30),
        (0.25, 0.75, -1.50,  1.00),
        (0.30, 0.70, -0.50, -0.50),
        (0.30, 0.70, -1.00,  0.50),
        (0.30, 0.70, -1.50, -0.50),
        (0.30, 0.70, -0.80, -1.00),
        (0.25, 0.75, -1.00,  0.80),
        (0.28, 0.72, -0.90,  0.20),
        (0.32, 0.68, -1.10, -0.30),
        (0.30, 0.70, -1.00, -0.50),
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Precompute true integrals using quad
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
            result, _ = quad(integrand_u, 0, 1, limit=200)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))

    n_design = len(true_vals)
    true_vals_arr = np.array([tv[5] for tv in true_vals])
    cosmos_params = np.array([tv[:4] for tv in true_vals])
    z_max_arr = np.array([tv[4] for tv in true_vals])

    def _compute_inv_E(nodes, cosmos_params, z_max_arr):
        """Compute 1/E for all design points and nodes. Returns (n_design, 8) array."""
        z_eval = z_max_arr[:, None] * nodes[None, :]
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        exponent = -3.0 * cosmos_params[:, 3:4] * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + cosmos_params[:, 2:3])) * np.exp(exponent)
        E2 = cosmos_params[:, 0:1] / a3 + cosmos_params[:, 1:2] * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        return 1.0 / E

    def _worst_rel_error(nodes, weights):
        """Compute worst relative error in percent."""
        inv_E = _compute_inv_E(nodes, cosmos_params, z_max_arr)
        approx = z_max_arr * (inv_E @ weights)
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        return np.max(rel_err)

    def _solve_weights_lp(nodes):
        """Solve for optimal weights via linear programming to minimize max relative error."""
        inv_E = _compute_inv_E(nodes, cosmos_params, z_max_arr)
        A = z_max_arr[:, None] * inv_E  # (n_design, 8)
        b_true = true_vals_arr

        # Minimize t s.t. -t <= (A @ w - b_true)/b_true <= t, sum(w)=1, w>=0
        # Variables: w_0..w_7, t
        # Constraints:
        #   A @ w - b_true - b_true * t <= 0   (for each j)
        #   b_true * t - A @ w + b_true <= 0   (for each j)
        #   sum(w) = 1
        #   w_i >= 0, t >= 0

        n = 8
        # c: objective = t (index 8)
        c = np.zeros(n + 1)
        c[n] = 1.0

        # Inequality constraints: A_ub @ x <= b_ub
        # Row j: A[j,:] @ w - b_true[j]*t <= b_true[j]
        # Row n+j: -A[j,:] @ w + b_true[j]*t <= b_true[j]
        A_ub = np.zeros((2 * n_design, n + 1))
        b_ub = np.zeros(2 * n_design)
        for j in range(n_design):
            A_ub[j, :n] = A[j, :]
            A_ub[j, n] = -b_true[j]
            b_ub[j] = b_true[j]
            A_ub[n_design + j, :n] = -A[j, :]
            A_ub[n_design + j, n] = b_true[j]
            b_ub[n_design + j] = b_true[j]

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w_i >= 0, t >= 0
        bounds = [(0, None)] * n + [(0, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            w = res.x[:n]
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            return w, res.x[n]
        else:
            # Fallback: equal weights
            return np.ones(n) / n, 1e10

    def _optimize_nodes_given_weights(weights):
        """Optimize nodes given fixed weights using Nelder-Mead."""
        def obj_nodes(raw):
            nodes = 0.5 + 0.5 * np.tanh(raw)
            if np.any(np.diff(nodes) <= 1e-14):
                return 1e10
            inv_E = _compute_inv_E(nodes, cosmos_params, z_max_arr)
            approx = z_max_arr * (inv_E @ weights)
            rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
            return np.max(rel_err)

        gl_nodes, _ = np.polynomial.legendre.leggauss(8)
        gl_01 = 0.5 * (gl_nodes + 1.0)
        raw0 = np.arctanh(2.0 * gl_01 - 1.0)

        try:
            res = minimize(obj_nodes, raw0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.success:
                nodes = 0.5 + 0.5 * np.tanh(res.x)
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                order = np.argsort(nodes)
                nodes = nodes[order]
                return nodes
        except Exception:
            pass
        return gl_01.copy()

    # Phase 1: Alternating optimization
    # Start with Gauss-Legendre nodes
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    current_nodes = 0.5 * (gl_nodes + 1.0)
    current_weights = gl_weights.copy()

    best_nodes = current_nodes.copy()
    best_weights = current_weights.copy()
    best_err = _worst_rel_error(best_nodes, best_weights)

    for iteration in range(12):
        if time.time() > deadline - 10:
            break

        # Step 1: Optimize weights given current nodes (LP)
        new_weights, lp_err = _solve_weights_lp(current_nodes)
        if np.all(new_weights > 0):
            current_weights = new_weights
            err_w = _worst_rel_error(current_nodes, current_weights)
            if err_w < best_err:
                best_err = err_w
                best_nodes = current_nodes.copy()
                best_weights = current_weights.copy()

        # Step 2: Optimize nodes given current weights
        new_nodes = _optimize_nodes_given_weights(current_weights)
        current_nodes = new_nodes
        err_n = _worst_rel_error(current_nodes, current_weights)
        if err_n < best_err:
            best_err = err_n
            best_nodes = current_nodes.copy()
            best_weights = current_weights.copy()

        if time.time() > deadline - 10:
            break

        # Step 3: Re-solve weights with updated nodes
        new_weights, lp_err2 = _solve_weights_lp(current_nodes)
        if np.all(new_weights > 0):
            current_weights = new_weights
            err_w2 = _worst_rel_error(current_nodes, current_weights)
            if err_w2 < best_err:
                best_err = err_w2
                best_nodes = current_nodes.copy()
                best_weights = current_weights.copy()

    # Phase 2: Joint refinement with Nelder-Mead on (nodes, log_weights)
    def _joint_objective(params):
        raw = params[:8]
        log_w = params[8:]
        nodes = 0.5 + 0.5 * np.tanh(raw)
        if np.any(np.diff(nodes) <= 1e-14):
            return 1e10
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-300:
            return 1e10
        weights = weights / w_sum
        return _worst_rel_error(nodes, weights)

    def _joint_hard(params):
        raw = params[:8]
        log_w = params[8:]
        nodes = 0.5 + 0.5 * np.tanh(raw)
        if np.any(np.diff(nodes) <= 1e-14):
            return 1e10
        weights = np.exp(log_w)
        weights = weights / np.sum(weights)
        return _worst_rel_error(nodes, weights)

    # Initialize from best found
    init_raw = np.arctanh(2.0 * best_nodes - 1.0)
    init_log_w = np.log(np.maximum(best_weights, 1e-15))
    x0 = np.concatenate([init_raw, init_log_w])

    best_params = x0.copy()
    best_joint_err = _joint_hard(x0)

    if time.time() < deadline - 15:
        try:
            res = minimize(_joint_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            h = _joint_hard(res.x)
            if h < best_joint_err:
                best_joint_err = h
                best_params = res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 10:
        try:
            res2 = minimize(_joint_objective, best_params, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-14})
            h = _joint_hard(res2.x)
            if h < best_joint_err:
                best_joint_err = h
                best_params = res2.x.copy()
        except Exception:
            pass

    # Extract final result
    raw = best_params[:8]
    log_w = best_params[8:]
    nodes = 0.5 + 0.5 * np.tanh(raw)
    weights = np.exp(log_w)
    weights = weights / np.sum(weights)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    # Compare with alternating result
    final_err = _worst_rel_error(nodes, weights)
    alt_err = _worst_rel_error(best_nodes, best_weights)
    if alt_err < final_err:
        nodes, weights = best_nodes, best_weights
        final_err = alt_err

    # Final check and normalization
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
