"""Combined approach: Chebyshev init + differential evolution for nodes + Remez equioscillation polish."""
import time
import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution

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

# High-accuracy reference quadrature
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    if z < 1e-10:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _compute_F_T(nodes, test_cosmos, test_zs):
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
    """Solve min t s.t. |F w - T| <= t, sum(w)=1, w>=0. Returns (t, w)."""
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
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[-1], res.x[:n_vars]
        else:
            return 1e10, np.ones(8) / 8.0
    except Exception:
        return 1e10, np.ones(8) / 8.0

def _solve_weights_ls(F, T):
    """Weighted least squares with positivity and sum-to-1 constraints."""
    W = 1.0 / (T + 1e-30)
    W[W > 1e10] = 0
    Fw = F * W[:, None]
    Tw = T * W

    def obj(w):
        r = Fw @ w - Tw
        return np.dot(r, r)

    def grad(w):
        r = Fw @ w - Tw
        return 2.0 * Fw.T @ r

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
             'jac': lambda w: np.ones(8)}]
    bounds = [(0, None)] * 8
    try:
        res = minimize(obj, np.ones(8)/8.0, method='SLSQP', jac=grad,
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-14})
        if res.success:
            w = np.maximum(res.x, 0)
            w = w / np.sum(w)
            return w
    except Exception:
        pass
    return np.ones(8) / 8.0

def _chebyshev_nodes(n=8):
    """Chebyshev nodes on [0,1] (clustering near endpoints)."""
    k = np.arange(n)
    x = 0.5 * (1.0 - np.cos(np.pi * (2*k + 1) / (2*n)))
    return np.sort(x)

def _decode_nodes(params):
    nodes_raw = 1.0 / (1.0 + np.exp(-params))
    nodes = np.sort(nodes_raw)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-7:
            nodes[i] = nodes[i-1] + 1e-7
    return nodes

def _encode_nodes(nodes):
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    return -np.log(1.0/nodes - 1.0)

# Test cosmologies for node search (smaller set for speed)
_SEARCH_COSMOS = [
    (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
    (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
    (-0.2, 0.2), (0.8, -0.8)
]
_SEARCH_ZS = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])

# Full test set for final weight solve
_FINAL_COSMOS = [
    (0.0, 0.0), (-1.0, 0.0), (0.5, -1.0), (-1.5, 0.5),
    (-0.5, 0.5), (0.3, -0.3), (-2.0, 1.0), (1.0, -2.0),
    (-0.2, 0.2), (0.8, -0.8), (0.1, 0.1), (-0.1, -0.1),
    (0.2, -0.5), (-0.8, 0.2), (-1.2, 0.8), (0.6, -1.2)
]
_FINAL_ZS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

def _node_objective(params):
    if _time_left() < 2.0:
        return 1e10
    nodes = _decode_nodes(params)
    F, T = _compute_F_T(nodes, _SEARCH_COSMOS, _SEARCH_ZS)
    t, _ = _solve_weights_lp(F, T)
    return t

def _remez_polish(nodes, cosmos, zs, max_iter=30):
    """
    Remez-like iteration: alternate between finding worst-error test points
    and solving LP for weights that equioscillate.
    """
    if _time_left() < 3.0:
        return nodes, None

    # Start with all test points
    all_indices = []
    for i, (w0, wa) in enumerate(cosmos):
        for j, z in enumerate(zs):
            all_indices.append((i, j))

    # Current active set
    active = list(all_indices)

    best_t = 1e10
    best_w = np.ones(8) / 8.0

    for it in range(max_iter):
        if _time_left() < 2.0:
            break

        # Build F, T for active set
        n_act = len(active)
        F_act = np.zeros((n_act, 8))
        T_act = np.zeros(n_act)
        for k, (ci, zi) in enumerate(active):
            w0, wa = cosmos[ci]
            z = zs[zi]
            if z < 1e-10:
                F_act[k, :] = 1.0
                T_act[k] = 1.0
            else:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = _E_cpl(a_vals, w0, wa)
                F_act[k, :] = 1.0 / E_vals
                true_val = _true_dL(z, w0, wa)
                T_act[k] = true_val / ((1.0 + z) * z)

        # Solve LP for weights
        t, w = _solve_weights_lp(F_act, T_act)
        if t < best_t:
            best_t = t
            best_w = w.copy()

        # Find worst-error point outside active set
        if len(active) < len(all_indices):
            # Compute errors for all points
            F_all = np.zeros((len(all_indices), 8))
            T_all = np.zeros(len(all_indices))
            for k, (ci, zi) in enumerate(all_indices):
                w0, wa = cosmos[ci]
                z = zs[zi]
                if z < 1e-10:
                    F_all[k, :] = 1.0
                    T_all[k] = 1.0
                else:
                    a_vals = 1.0 / (1.0 + z * nodes)
                    E_vals = _E_cpl(a_vals, w0, wa)
                    F_all[k, :] = 1.0 / E_vals
                    true_val = _true_dL(z, w0, wa)
                    T_all[k] = true_val / ((1.0 + z) * z)

            errors = np.abs(F_all @ w - T_all) / (T_all + 1e-30)
            active_set = set(active)
            max_err = t
            worst_idx = -1
            for k in range(len(all_indices)):
                if all_indices[k] not in active_set:
                    if errors[k] > max_err:
                        max_err = errors[k]
                        worst_idx = k

            if worst_idx >= 0 and max_err > t * 1.001:
                active.append(all_indices[worst_idx])
                # Remove the least important active point (smallest error)
                if len(active) > 20:
                    act_errors = np.abs(F_act @ w - T_act) / (T_act + 1e-30)
                    min_idx = np.argmin(act_errors)
                    active.pop(min_idx)
            else:
                # No improvement, stop
                break
        else:
            # All points active, check convergence
            if it > 5 and t < 1e-12:
                break

    return nodes, best_w

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0

    # Phase 1: Chebyshev initialization + differential evolution for nodes
    cheb_nodes = _chebyshev_nodes(8)
    params_init = _encode_nodes(cheb_nodes)

    best_params = params_init.copy()
    best_t = _node_objective(best_params)

    # Differential evolution for global search
    if _time_left() > 40.0:
        bounds_de = [(-10, 10)] * 8
        try:
            de_result = differential_evolution(
                _node_objective, bounds_de,
                seed=42, maxiter=30, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0),
                recombination=0.9, polish=False,
                init='sobol'
            )
            if de_result.fun < best_t:
                best_t = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Local Nelder-Mead polish from multiple starting points
    if _time_left() > 25.0:
        np.random.seed(42)
        starting_points = [best_params.copy()]
        for _ in range(4):
            starting_points.append(best_params + np.random.randn(8) * 0.3)

        for sp in starting_points:
            if _time_left() < 8.0:
                break
            try:
                res = minimize(_node_objective, sp, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-9,
                                       'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_t:
                    best_t = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    nodes = _decode_nodes(best_params)

    # Phase 3: Remez-like equioscillation polish
    if _time_left() > 10.0:
        nodes, remez_w = _remez_polish(nodes, _FINAL_COSMOS, _FINAL_ZS, max_iter=40)
        if remez_w is not None and _time_left() > 5.0:
            # After Remez, do a final local node refinement
            params_refined = _encode_nodes(nodes)
            try:
                res_final = minimize(_node_objective, params_refined,
                                    method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-10,
                                             'fatol': 1e-13, 'adaptive': True})
                if res_final.fun < best_t:
                    best_t = res_final.fun
                    best_params = res_final.x.copy()
                    nodes = _decode_nodes(best_params)
                    nodes, remez_w = _remez_polish(nodes, _FINAL_COSMOS,
                                                    _FINAL_ZS, max_iter=20)
            except Exception:
                pass

    # Phase 4: Final weight solve with full test set
    F_final, T_final = _compute_F_T(nodes, _FINAL_COSMOS, _FINAL_ZS)
    t_final, weights = _solve_weights_lp(F_final, T_final)

    if weights is None or not np.all(np.isfinite(weights)):
        weights = _solve_weights_ls(F_final, T_final)

    weights = np.abs(weights)
    wsum = weights.sum()
    if wsum < 1e-15:
        weights = np.ones(8) / 8.0
    else:
        weights = weights / wsum

    # Ensure nodes are strictly increasing in [0,1]
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8

    return {"nodes": nodes, "weights": weights}
