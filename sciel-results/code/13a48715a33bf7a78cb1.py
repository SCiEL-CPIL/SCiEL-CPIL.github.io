"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np


def _cpl_g(x, w0, wa):
    """Model-dependent factor g(x)=1/E(1-x) for CPL. x = z/(1+z), a=1-x."""
    a = 1.0 - x
    wa_eff = wa / a
    g = np.power(a, 3.0 * (1.0 + w0 + wa_eff))
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def _cpl_g_log(x, w0, wa):
    """Log of g for numerical stability."""
    a = np.clip(1.0 - x, 1e-12, 1.0)
    return 3.0 * (1.0 + w0 + wa / a) * np.log(a)


def _project_positive(w, tol=1e-10):
    """Project weights onto the cone w_i>0, sum w = 1 (Euclidean projection)."""
    w = np.asarray(w, dtype=float)
    n = w.size
    for _ in range(500):
        lam = max(0.0, (w.sum() - 1.0) / n)
        w_new = np.maximum(w - lam, tol)
        if np.abs(w_new.sum() - 1.0) < 1e-13 and np.all(w_new >= tol):
            w = w_new
            break
        w = w_new
    w = np.maximum(w, tol)
    w = w / w.sum()
    return w


def _worst_rel_error(nodes, weights, samples, f_true):
    """Worst relative error in percent over the sample library."""
    Q = np.sum(weights[:, None] * samples[None, :], axis=0)
    err = np.abs(Q - f_true) / np.maximum(f_true, 1e-30) * 100.0
    return float(err.max())


def _solve(w0_grid, wa_grid):
    # Build model library
    n_models = len(w0_grid) * len(wa_grid)
    M = 4000
    x = np.linspace(1e-4, 1.0 - 1e-4, M)
    f_true = np.empty(n_models)
    lib = np.empty((n_models, M))
    k = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            g = _cpl_g(x, w0, wa)
            f = x * g
            lib[k] = f
            f_true[k] = np.trapz(f, x)
            k += 1
    f_true = np.maximum(f_true, 1e-30)

    # Weighting: emphasize regions where relative error matters (small f -> large rel err)
    # Use a moderate inverse weighting to balance, but keep it bounded.
    wgt = 1.0 / np.sqrt(np.maximum(f_true, 1e-3))
    wgt = wgt / wgt.max()

    A = lib * wgt[:, None]          # (m, M)
    b = f_true * wgt                # (m,)

    # Initial nodes: composite midpoint (the seed)
    nodes = (np.arange(8) + 0.5) / 8.0
    weights = np.full(8, 0.125)

    # Alternating optimization
    for it in range(400):
        # --- Update weights: min max_j |sum_i w_i A_ij - b_j|  s.t. w>=0, sum w=1
        # S-procedure: min t s.t. |A^T w - b| <= t*wgt_max... use weighted form.
        # We minimize max_j |sum_i w_i A_ij - b_j|  (weighted already in A,b)
        A2 = A.T                    # (M, m)
        # Variables: w (8), t (1). Constraints: A2 w - t*1 <= b ; -(A2 w) - t*1 <= b
        c = np.zeros(9)
        c[8] = 1.0
        c_eq = np.zeros(8)
        c_eq[:] = 1.0
        A_ub = np.zeros((2 * M, 9))
        b_ub = np.zeros(2 * M)
        A_ub[:M, :8] = A2
        A_ub[:M, 8] = -1.0
        A_ub[M:, :8] = -A2
        A_ub[M:, 8] = -1.0
        b_ub[:M] = b
        b_ub[M:] = b
        A_eq = np.zeros((1, 8))
        A_eq[0, :] = 1.0
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * 8 + [(0.0, None)]
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            weights = _project_positive(res.x[:8])
        else:
            # fallback: least squares with positivity
            w_ls, *_ = np.linalg.lstsq(A2, b, rcond=None)
            weights = _project_positive(w_ls)

        # --- Update nodes: minimize worst relative error (unweighted)
        def objective(v):
            nd = np.sort(v)
            s = lib @ (nd * 0 + 0)  # placeholder
            # build samples at nodes
            S = np.empty((n_models, 8))
            for i in range(8):
                S[:, i] = _cpl_g(nd[i], 0, 0)  # dummy, recompute below
            # recompute properly
            S = np.array([_cpl_g(nd, w0, wa) for w0, wa in zip(*np.meshgrid(w0_grid, wa_grid, indexing='ij')).T])
            S = S.reshape(n_models, 8) * nd
            Q = S @ weights
            err = np.abs(Q - f_true) / f_true
            return err.max()

        from scipy.optimize import minimize
        x0 = nodes.copy()
        cons = [{"type": "ineq", "fun": lambda v: v[0] - 1e-3},
                {"type": "ineq", "fun": lambda v: 1 - 1e-3 - v[-1]}]
        res2 = minimize(objective, x0, method="L-BFGS-B",
                        bounds=[(1e-3, 1 - 1e-3)] * 8, options={"maxiter": 200, "ftol": 1e-12})
        if res2.success:
            nodes = np.sort(res2.x)
        # enforce strict increase and bounds
        nodes = np.clip(nodes, 1e-3, 1 - 1e-3)
        # ensure strictly increasing
        for i in range(1, 8):
            if nodes[i] <= nodes[i - 1] + 1e-6:
                nodes[i] = nodes[i - 1] + 1e-6
        nodes[-1] = min(nodes[-1], 1 - 1e-3)

    # Final: re-project weights and polish nodes with a finer local search
    weights = _project_positive(weights)

    # Refine nodes via Nelder-Mead on the worst relative error
    def obj_nodes(v):
        nd = np.sort(np.clip(v, 1e-3, 1 - 1e-3))
        for i in range(1, 8):
            if nd[i] <= nd[i - 1] + 1e-6:
                nd[i] = nd[i - 1] + 1e-6
        S = np.array([_cpl_g(nd, w0, wa) for w0, wa in
                      zip(*np.meshgrid(w0_grid, wa_grid, indexing='ij')).T]).reshape(n_models, 8) * nd
        Q = S @ weights
        return (np.abs(Q - f_true) / f_true).max()

    from scipy.optimize import minimize
    res3 = minimize(obj_nodes, nodes, method="Nelder-Mead",
                    options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-14})
    if res3.success:
        nodes = np.sort(res3.x)
        nodes = np.clip(nodes, 1e-3, 1 - 1e-3)
        for i in range(1, 8):
            if nodes[i] <= nodes[i - 1] + 1e-6:
                nodes[i] = nodes[i - 1] + 1e-6

    # Final weight projection
    weights = _project_positive(weights)

    # Final sanity: ensure sum exactly 1 and positivity
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    nodes = np.clip(np.sort(nodes), 1e-3, 1 - 1e-3)

    return {"nodes": nodes, "weights": weights}


def construct():
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    try:
        rule = _solve(w0_grid, wa_grid)
    except Exception:
        # Fallback: composite midpoint
        n = 8
        rule = {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    # Enforce contract: strictly increasing in [0,1], positive weights summing to 1.
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
