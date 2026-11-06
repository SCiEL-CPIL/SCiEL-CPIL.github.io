"""Universal 8-node positive quadrature for dimensionless luminosity distance.

Strategy:
- The luminosity distance D_L/(c/H0) = (1+z)*z * int_0^1 dt/E(z*t)
  is approximated by (1+z)*z * sum_i w_i / E(z*x_i).
- For a fixed set of 8 nodes x_i, the optimal positive weights minimizing
  worst-case relative error over a training set are found via a small LP.
- Nodes are optimized using Nelder-Mead on a smooth log-sum-exp surrogate
  of the minimax relative error, with logit parameterization to handle
  monotonicity constraints.
- Multiple diverse initial node sets are tried; the best is refined.
- Strict wall-clock deadline guard with fallback to Gauss-Legendre.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre

_T0 = time.time()
_DEADLINE = _T0 + 82.0  # safety margin before 90s kill


def _time_left():
    return _DEADLINE - time.time()


def _E_flat(z, om_m, w0, wa):
    """E(z) for spatially flat CPL: Om(1+z)^3 + Ode(1+z)^(3+3w0+3wa z)."""
    z = np.asarray(z, dtype=float)
    opz = 1.0 + z
    dm = om_m * opz ** 3
    de = (1.0 - om_m) * opz ** (3.0 + 3.0 * w0 + 3.0 * wa * z)
    return np.sqrt(dm + de)


def _ref_integral(z, om_m, w0, wa):
    """High-accuracy reference integral of 1/E(z*t) dt from 0 to 1 using quad."""
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
    Build a dense training set spanning z and (w0, wa).
    Returns list of (z, om_m, w0, wa).
    """
    z_vals = np.concatenate([
        np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0,
                  1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]),
        np.logspace(np.log10(0.01), np.log10(10.0), 40)
    ])
    z_vals = np.unique(np.sort(z_vals))

    om_m_vals = [0.25, 0.30, 0.35]
    w0_vals = [-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5]
    wa_vals = [-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0]

    params = []
    # All z with wa=0
    for z in z_vals:
        for om_m in om_m_vals:
            for w0 in w0_vals:
                params.append((z, om_m, w0, 0.0))
    # All z with w0=0
    for z in z_vals:
        for om_m in om_m_vals:
            for wa in wa_vals:
                if wa != 0.0:
                    params.append((z, om_m, 0.0, wa))
    # Interior grid at key z
    for z in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for om_m in [0.27, 0.30, 0.33]:
            for w0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for wa in [-0.5, 0.0, 0.5]:
                    params.append((z, om_m, w0, wa))

    # Deduplicate
    seen = set()
    unique_params = []
    for p in params:
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6), round(p[3], 6))
        if key not in seen:
            seen.add(key)
            unique_params.append(p)

    # Limit size for tractability
    if len(unique_params) > 400:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(unique_params), 400, replace=False)
        unique_params = [unique_params[i] for i in idx]

    return unique_params


def _compute_A_and_refs(params, nodes):
    """
    For given nodes, compute A[i,j] = 1/E(z_i * nodes_j) and refs[i].
    Returns A (N, 8) and refs (N,).
    """
    N = len(params)
    n = len(nodes)
    A = np.zeros((N, n))
    refs = np.zeros(N)

    for i, (z, om_m, w0, wa) in enumerate(params):
        if z < 1e-14:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_flat(zt, om_m, w0, wa)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, om_m, w0, wa)

    return A, refs


def _minimax_error(nodes, params):
    """Compute minimax relative error for given nodes."""
    A, refs = _compute_A_and_refs(params, nodes)
    w, t = _solve_weights_lp(A, refs, len(nodes))
    if w is None:
        return np.inf
    return t


def _project_nodes(nodes):
    """Project nodes to be strictly increasing in (0,1)."""
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    nodes = np.sort(nodes)
    min_sep = 1e-8
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < min_sep:
            nodes[i] = nodes[i - 1] + min_sep
    nodes[-1] = min(nodes[-1], 1 - 1e-6)
    return nodes


def _logit(x):
    """Logit transform: maps (0,1) to (-inf, inf)."""
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return np.log(x / (1 - x))


def _expit(x):
    """Inverse logit: maps (-inf, inf) to (0,1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _smooth_minimax_surrogate(theta, params, temperature=0.01):
    """
    Smooth log-sum-exp surrogate of the minimax relative error.
    theta: 7-dimensional vector of logit-differences (monotonicity enforced).
    """
    n = 8
    # Reconstruct nodes from logit parameterization
    # theta[0] = logit(x_0), theta[i] = logit(x_i) - logit(x_{i-1}) for i>=1
    logits = np.zeros(n)
    logits[0] = theta[0]
    for i in range(1, n):
        logits[i] = logits[i - 1] + theta[i]
    nodes = _expit(logits)
    nodes = _project_nodes(nodes)
    
    if not np.all(np.diff(nodes) > 1e-8):
        return 1e6
    
    A, refs = _compute_A_and_refs(params, nodes)
    w, t = _solve_weights_lp(A, refs, n)
    if w is None:
        return 1e6
    
    # Compute individual relative errors
    errors = np.zeros(len(params))
    for i in range(len(params)):
        quad_val = np.dot(w, A[i, :])
        if refs[i] > 1e-14:
            errors[i] = abs(quad_val - refs[i]) / refs[i]
        else:
            errors[i] = abs(quad_val - refs[i])
    
    # Log-sum-exp for smooth minimax
    max_e = np.max(errors)
    if max_e < 1e-15:
        return 0.0
    lse = max_e + temperature * np.log(np.sum(np.exp((errors - max_e) / temperature)))
    return lse


def _optimize_nodes_logit(params, initial_nodes, max_iter=200, temperature=0.01):
    """
    Optimize nodes using Nelder-Mead on logit-parameterized smooth minimax surrogate.
    """
    n = len(initial_nodes)
    
    # Convert initial nodes to logit parameterization
    logits = _logit(initial_nodes)
    theta0 = np.zeros(n - 1)
    theta0[0] = logits[0]
    for i in range(1, n):
        theta0[i] = logits[i] - logits[i - 1]
    
    def objective(theta):
        return _smooth_minimax_surrogate(theta, params, temperature)
    
    try:
        res = minimize(objective, theta0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-12,
                                'fatol': 1e-12, 'adaptive': True})
        # Reconstruct nodes
        logits = np.zeros(n)
        logits[0] = res.x[0]
        for i in range(1, n):
            logits[i] = logits[i - 1] + res.x[i]
        nodes = _expit(logits)
        nodes = _project_nodes(nodes)
        if np.all(np.diff(nodes) > 1e-8):
            return nodes
    except Exception:
        pass
    return initial_nodes


def _get_weights_for_nodes(nodes, params):
    """Get optimal weights for given nodes."""
    A, refs = _compute_A_and_refs(params, nodes)
    w, t = _solve_weights_lp(A, refs, len(nodes))
    if w is None:
        return np.full(len(nodes), 1.0 / len(nodes))
    return w


def _fallback():
    n = 8
    gl_n, gl_w = roots_legendre(n)
    nodes = (gl_n + 1.0) / 2.0
    weights = gl_w / 2.0
    return nodes, weights


def _make_rule():
    try:
        if _time_left() < 5.0:
            return _fallback()

        params = _build_training()

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
            cand = _project_nodes(cand)
            if np.all(np.diff(cand) > 1e-6):
                candidates.append(cand)

        # 4. Log-spaced
        for p in [1.3, 1.5, 2.0, 2.5]:
            t = np.linspace(0, 1, 8)
            ln = 1.0 - (1.0 - t) ** p
            ln = _project_nodes(ln)
            if np.all(np.diff(ln) > 1e-6):
                candidates.append(ln)

        # 5. Power-law
        for p in [0.7, 0.8, 1.0, 1.2, 1.5]:
            t = np.linspace(0, 1, 8)
            pn = t ** p
            pn = _project_nodes(pn)
            if np.all(np.diff(pn) > 1e-6):
                candidates.append(pn)

        # Evaluate initial candidates
        best_nodes = None
        best_err = np.inf
        for cand in candidates:
            if _time_left() < 5.0:
                break
            err = _minimax_error(cand, params)
            if err < best_err:
                best_err = err
                best_nodes = cand.copy()

        if best_nodes is None:
            return _fallback()

        # Refine with logit-parameterized Nelder-Mead on smooth surrogate
        if _time_left() > 15.0:
            refined = _optimize_nodes_logit(params, best_nodes, max_iter=150, temperature=0.01)
            refined_err = _minimax_error(refined, params)
            if refined_err < best_err:
                best_nodes = refined
                best_err = refined_err

        # Second round with lower temperature for sharper minimax
        if _time_left() > 10.0:
            refined2 = _optimize_nodes_logit(params, best_nodes, max_iter=100, temperature=0.003)
            refined_err2 = _minimax_error(refined2, params)
            if refined_err2 < best_err:
                best_nodes = refined2
                best_err = refined_err2

        # Get weights
        weights = _get_weights_for_nodes(best_nodes, params)

        # Validate
        nodes = _project_nodes(best_nodes)
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


_NODES, _WEIGHTS = _make_rule()


def construct():
    return {"nodes": _NODES, "weights": _WEIGHTS}
