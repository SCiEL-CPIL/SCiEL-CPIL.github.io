"""Universal 8-node positive quadrature for dimensionless luminosity distance.

Strategy:
- Represent the integrand 1/E(a) with a compact basis of power functions of a
  (a = 1/(1+z*t)) that spans the CPL family for spatially flat models.
- Precompute basis values at candidate nodes for a dense z-grid.
- Solve a small LP for weights that minimize worst relative error over the
  training set, then refine nodes with Nelder-Mead on the resulting minimax
  error. Multiple starting node sets are used to avoid local minima.
- Strict wall-clock deadline guard with fallback to Gauss-Legendre nodes.
"""

import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre

_T0 = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill


def _time_left():
    return _DEADLINE - time.time()


def _E_flat(z, om_m, w0, wa):
    """E(z) for spatially flat CPL: Om(1+z)^3 + Ode(1+z)^(3+3w0+3wa z)."""
    om_m = np.asarray(om_m, dtype=float)
    opz = 1.0 + z
    dm = om_m * opz ** 3
    de = (1.0 - om_m) * opz ** (3.0 + 3.0 * w0 + 3.0 * wa * z)
    return np.sqrt(dm + de)


def _integrand(z, t, om_m, w0, wa):
    """1/E(z*t) for spatially flat CPL model."""
    return 1.0 / _E_flat(z * t, om_m, w0, wa)


def _basis_vals(z, t, om_m, w0, wa, powers):
    """
    Evaluate basis functions at z*t for given powers.
    Basis: a^p for p in powers, where a = 1/(1+z*t).
    Returns array of shape (len(z), len(t), len(powers)).
    """
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    a = 1.0 / (1.0 + np.outer(z, t))  # (nz, nt)
    vals = np.empty((a.shape[0], a.shape[1], len(powers)))
    for k, p in enumerate(powers):
        vals[:, :, k] = a ** p
    return vals


def _ref_integral(z, om_m, w0, wa):
    """High-accuracy reference integral of 1/E(z*t) dt from 0 to 1 using quad."""
    from scipy.integrate import quad
    if z < 1e-14:
        return 1.0
    f = lambda t: 1.0 / _E_flat(z * t, om_m, w0, wa)
    val, _ = quad(f, 0, 1, limit=200, epsabs=1e-13, epsrel=1e-13)
    return val


def _solve_weights_lp(A, refs, n):
    """
    Solve LP: minimize t s.t.
      A_i @ w - t*ref_i <= ref_i
     -A_i @ w - t*ref_i <= ref_i
      sum(w) = 1, w >= 0, t >= 0
    Returns (weights, t) or (None, inf).
    """
    N = A.shape[0]
    c = np.zeros(n + 1)
    c[n] = 1.0

    A_ub = np.zeros((2 * N, n + 1))
    b_ub = np.zeros(2 * N)
    for i in range(N):
        A_ub[2 * i, :n] = A[i, :]
        A_ub[2 * i, n] = -refs[i]
        b_ub[2 * i] = refs[i]
        A_ub[2 * i + 1, :n] = -A[i, :]
        A_ub[2 * i + 1, n] = -refs[i]
        b_ub[2 * i + 1] = -refs[i]

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * (n + 1)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            w = res.x[:n]
            s = w.sum()
            if s > 1e-15:
                w /= s
            return w, res.x[n]
    except Exception:
        pass
    return None, np.inf


def _build_training():
    """
    Build a dense training set:
    - z: log-spaced from 0.01 to 10
    - om_m: 0.25, 0.30, 0.35
    - w0, wa: representative grid
    Returns z_vals, params list, refs array.
    """
    z_vals = np.logspace(np.log10(0.01), np.log10(10.0), 60)
    om_m_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]

    params = []
    for om_m in om_m_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                params.append((om_m, w0, wa))

    # Add some intermediate points
    for om_m in [0.27, 0.32]:
        for w0 in [-0.8, -0.2, 0.3]:
            for wa in [-0.3, 0.3]:
                params.append((om_m, w0, wa))

    # Deduplicate
    seen = set()
    unique_params = []
    for p in params:
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        if key not in seen:
            seen.add(key)
            unique_params.append(p)

    return z_vals, unique_params


def _compute_A_and_refs(z_vals, params, nodes):
    """
    For given nodes, compute A[i,j] = 1/E(z_i * nodes_j) and refs[i].
    Returns A (N, 8) and refs (N,).
    """
    N = len(z_vals)
    n = len(nodes)
    A = np.zeros((N, n))
    refs = np.zeros(N)

    for i, z in enumerate(z_vals):
        for j, node in enumerate(nodes):
            A[i, j] = _integrand(z, node, params[0][0], params[0][1], params[0][2])
        # Actually, A depends on the model too. We need to structure this differently.
        # Let's compute per model.
    
    # Restructure: A is (N_z * N_models, n)
    N_total = len(z_vals) * len(params)
    A = np.zeros((N_total, n))
    refs = np.zeros(N_total)
    idx = 0
    for z in z_vals:
        for (om_m, w0, wa) in params:
            for j, node in enumerate(nodes):
                A[idx, j] = _integrand(z, node, om_m, w0, wa)
            refs[idx] = _ref_integral(z, om_m, w0, wa)
            idx += 1

    return A, refs


def _minimax_error(nodes, z_vals, params):
    """Compute minimax relative error for given nodes."""
    A, refs = _compute_A_and_refs(z_vals, params, nodes)
    w, t = _solve_weights_lp(A, refs, len(nodes))
    if w is None:
        return np.inf
    return t


def _optimize_nodes(z_vals, params, initial_nodes, max_iter=300):
    """
    Refine nodes using Nelder-Mead to minimize minimax error.
    """
    n = len(initial_nodes)

    def objective(nodes):
        # Ensure nodes are in (0,1) and increasing
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        nodes = np.sort(nodes)
        if not np.all(np.diff(nodes) > 1e-8):
            return 1e6
        return _minimax_error(nodes, z_vals, params)

    try:
        res = minimize(objective, initial_nodes, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10,
                                'fatol': 1e-10, 'adaptive': True})
        nodes = np.clip(res.x, 1e-6, 1 - 1e-6)
        nodes = np.sort(nodes)
        if not np.all(np.diff(nodes) > 1e-8):
            return initial_nodes
        return nodes
    except Exception:
        return initial_nodes


def _get_weights_for_nodes(nodes, z_vals, params):
    """Get optimal weights for given nodes."""
    A, refs = _compute_A_and_refs(z_vals, params, nodes)
    w, t = _solve_weights_lp(A, refs, len(nodes))
    if w is None:
        return np.full(len(nodes), 1.0 / len(nodes))
    return w


def _make_rule():
    try:
        if _time_left() < 5.0:
            return _fallback()

        z_vals, params = _build_training()

        # Initial node candidates
        candidates = []

        # 1. Gauss-Legendre
        gl_n, gl_w = roots_legendre(8)
        candidates.append((gl_n + 1.0) / 2.0)

        # 2. Chebyshev nodes
        k = np.arange(8)
        ch_n = np.cos((2 * k + 1) * np.pi / 16.0)
        candidates.append((ch_n + 1.0) / 2.0)

        # 3. Perturbed GL
        rng = np.random.RandomState(42)
        base_gl = (gl_n + 1.0) / 2.0
        for _ in range(10):
            pert = rng.uniform(-0.05, 0.05, 8)
            cand = base_gl + pert
            cand = np.clip(cand, 1e-4, 1 - 1e-4)
            cand = np.sort(cand)
            if np.all(np.diff(cand) > 1e-6):
                candidates.append(cand)

        # 4. Log-spaced
        for p in [1.5, 2.0, 2.5]:
            t = np.linspace(0, 1, 8)
            ln = 1.0 - (1.0 - t) ** p
            ln = np.clip(ln, 1e-4, 1 - 1e-4)
            if np.all(np.diff(ln) > 1e-6):
                candidates.append(ln)

        # 5. Power-law
        for p in [0.8, 1.0, 1.2, 1.5]:
            t = np.linspace(0, 1, 8)
            pn = t ** p
            pn = np.clip(pn, 1e-4, 1 - 1e-4)
            if np.all(np.diff(pn) > 1e-6):
                candidates.append(pn)

        # Evaluate initial candidates
        best_nodes = None
        best_err = np.inf
        for cand in candidates:
            err = _minimax_error(cand, z_vals, params)
            if err < best_err:
                best_err = err
                best_nodes = cand.copy()

        if best_nodes is None:
            return _fallback()

        # Refine with Nelder-Mead
        if _time_left() > 10.0:
            refined = _optimize_nodes(z_vals, params, best_nodes, max_iter=200)
            refined_err = _minimax_error(refined, z_vals, params)
            if refined_err < best_err:
                best_nodes = refined
                best_err = refined_err

        # Get weights
        weights = _get_weights_for_nodes(best_nodes, z_vals, params)

        # Validate
        nodes = np.sort(best_nodes)
        weights = np.asarray(weights, dtype=float)
        if (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))
                and np.all(np.diff(nodes) > 0)
                and nodes[0] >= 0.0 and nodes[-1] <= 1.0
                and np.all(weights > 0)):
            weights = weights / weights.sum()
            return nodes, weights

    except Exception:
        pass

    return _fallback()


def _fallback():
    n = 8
    gl_n, gl_w = roots_legendre(n)
    nodes = (gl_n + 1.0) / 2.0
    weights = gl_w / 2.0
    return nodes, weights


_NODES, _WEIGHTS = _make_rule()


def construct():
    return {"nodes": _NODES, "weights": _WEIGHTS}
