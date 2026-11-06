"""Universal 8-node positive quadrature for dimensionless luminosity distance.

Strategy:
- For a fixed set of 8 nodes, the optimal positive weights that minimize the
  worst-case relative error over a set of (z, w0, wa) test points are obtained
  by a small linear program (min t s.t. |A w - ref| <= t*ref, sum w = 1, w>=0).
- Nodes are optimized by a robust min-max approach: alternate between (a) finding
  the worst-case test point for the current (nodes, weights), and (b) taking a
  subgradient step on the nodes to reduce that worst-case error.
- Multiple diverse initial node sets are used; the best is refined.
- Strict wall-clock deadline guard with fallback to Gauss-Legendre.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre, roots_jacobi

_T0 = time.time()
_DEADLINE = _T0 + 85.0  # safety margin before 90s kill


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
    Build a dense training/validation set spanning z and (w0, wa).
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
    if len(unique_params) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(unique_params), 500, replace=False)
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
    # Ensure minimum separation
    min_sep = 1e-8
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < min_sep:
            nodes[i] = nodes[i - 1] + min_sep
    nodes[-1] = min(nodes[-1], 1 - 1e-6)
    return nodes


def _subgradient_node_update(nodes, weights, A, refs, params, lr=0.001):
    """
    Frank-Wolfe style subgradient update for nodes.
    """
    n = len(nodes)
    # Find worst-case points
    errors = np.zeros(len(params))
    for i in range(len(params)):
        quad_val = np.dot(weights, A[i, :])
        if refs[i] > 1e-14:
            errors[i] = abs(quad_val - refs[i]) / refs[i]
        else:
            errors[i] = abs(quad_val - refs[i])

    # Use top-K worst points for subgradient
    K = min(10, len(errors))
    top_idx = np.argsort(errors)[-K:]
    
    grad = np.zeros(n)
    for i in top_idx:
        z, om_m, w0, wa = params[i]
        if z < 1e-14 or refs[i] < 1e-14:
            continue
        quad_val = np.dot(weights, A[i, :])
        sign = 1.0 if quad_val > refs[i] else -1.0
        for j in range(n):
            t = nodes[j]
            zt = z * t
            opz = 1.0 + zt
            E = _E_flat(zt, om_m, w0, wa)
            if E < 1e-14:
                continue
            dE_dt = 0.0
            if om_m > 1e-14:
                dE_dt += 3.0 * om_m * z * opz ** 2
            exp_de = 3.0 + 3.0 * w0 + 3.0 * wa * zt
            if (1.0 - om_m) > 1e-14:
                dE_dt += (1.0 - om_m) * z * exp_de * opz ** (exp_de - 1.0)
            dE_dt = 0.5 * dE_dt / E
            df_dt = -dE_dt / (E * E)
            grad[j] += sign * weights[j] * df_dt / refs[i]
    
    nodes_new = nodes - lr * grad
    nodes_new = _project_nodes(nodes_new)
    return nodes_new


def _frank_wolfe_optimize_nodes(nodes, params, max_iter=50, lr_init=0.002):
    """
    Frank-Wolfe / subgradient optimization for nodes.
    """
    n = len(nodes)
    best_nodes = nodes.copy()
    best_err = _minimax_error(nodes, params)
    
    A, refs = _compute_A_and_refs(params, nodes)
    weights, _ = _solve_weights_lp(A, refs, n)
    if weights is None:
        weights = np.full(n, 1.0 / n)
    
    lr = lr_init
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        A, refs = _compute_A_and_refs(params, nodes)
        weights, t = _solve_weights_lp(A, refs, n)
        if weights is None:
            break
        if t < best_err:
            best_err = t
            best_nodes = nodes.copy()
        
        nodes_new = _subgradient_node_update(nodes, weights, A, refs, params, lr)
        
        new_err = _minimax_error(nodes_new, params)
        if new_err < best_err:
            nodes = nodes_new
            best_err = new_err
            best_nodes = nodes.copy()
        else:
            lr *= 0.8
        
        if lr < 1e-6:
            break
    
    return best_nodes, best_err


def _optimize_nodes_nelder_mead(z_params, initial_nodes, max_iter=100):
    """
    Refine nodes using Nelder-Mead to minimize minimax error.
    """
    n = len(initial_nodes)

    def objective(nodes):
        nodes = _project_nodes(nodes)
        if not np.all(np.diff(nodes) > 1e-8):
            return 1e6
        return _minimax_error(nodes, z_params)

    try:
        res = minimize(objective, initial_nodes, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10,
                                'fatol': 1e-10, 'adaptive': True})
        nodes = _project_nodes(res.x)
        if not np.all(np.diff(nodes) > 1e-8):
            return initial_nodes
        return nodes
    except Exception:
        return initial_nodes


def _get_weights_for_nodes(nodes, params):
    """Get optimal weights for given nodes."""
    A, refs = _compute_A_and_refs(params, nodes)
    w, t = _solve_weights_lp(A, refs, len(nodes))
    if w is None:
        return np.full(len(nodes), 1.0 / len(nodes))
    return w


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

        # 3. Gauss-Jacobi variants
        for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (1.5, 1.5)]:
            try:
                jn, _ = roots_jacobi(8, alpha, beta)
                jn_mapped = (jn + 1.0) / 2.0
                if np.all(np.diff(jn_mapped) > 1e-10):
                    candidates.append(jn_mapped)
            except:
                pass

        # 4. Perturbed GL
        rng = np.random.RandomState(42)
        base_gl = (gl_n + 1.0) / 2.0
        for _ in range(15):
            pert = rng.uniform(-0.05, 0.05, 8)
            cand = base_gl + pert
            cand = _project_nodes(cand)
            if np.all(np.diff(cand) > 1e-6):
                candidates.append(cand)

        # 5. Log-spaced
        for p in [1.3, 1.5, 2.0, 2.5]:
            t = np.linspace(0, 1, 8)
            ln = 1.0 - (1.0 - t) ** p
            ln = _project_nodes(ln)
            if np.all(np.diff(ln) > 1e-6):
                candidates.append(ln)

        # 6. Power-law
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
            err = _minimax_error(cand, params)
            if err < best_err:
                best_err = err
                best_nodes = cand.copy()

        if best_nodes is None:
            return _fallback()

        # Refine with Frank-Wolfe subgradient
        if _time_left() > 15.0:
            fw_nodes, fw_err = _frank_wolfe_optimize_nodes(best_nodes, params, max_iter=40, lr_init=0.003)
            if fw_err < best_err:
                best_nodes = fw_nodes
                best_err = fw_err

        # Further refine with Nelder-Mead
        if _time_left() > 10.0:
            refined = _optimize_nodes_nelder_mead(params, best_nodes, max_iter=80)
            refined_err = _minimax_error(refined, params)
            if refined_err < best_err:
                best_nodes = refined
                best_err = refined_err

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


def _fallback():
    n = 8
    gl_n, gl_w = roots_legendre(n)
    nodes = (gl_n + 1.0) / 2.0
    weights = gl_w / 2.0
    return nodes, weights


_NODES, _WEIGHTS = _make_rule()


def construct():
    return {"nodes": _NODES, "weights": _WEIGHTS}
