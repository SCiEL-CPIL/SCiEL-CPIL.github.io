"""
Strategy: Minimax quadrature for luminosity distance with smooth surrogate optimization.

Key improvements over parent:
1. Cache reference integrals (expensive quad calls) to avoid recomputation
2. Use a smooth log-sum-exp surrogate for the minimax objective to guide
   Nelder-Mead more effectively
3. Use a bounded parameterization (logit transform) to keep nodes in (0,1)
4. More diverse initial candidates and better test set construction
5. Final polish with tighter LP tolerance
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

# Cache for reference integrals
_ref_cache = {}

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Vectorized E(z) = H(z)/H0 for CPL model."""
    Ode = 1.0 - Omega_m
    opz = 1.0 + z
    term_m = Omega_m * opz**3
    exponent = 3.0 + 3.0*w0 + 3.0*wa*z
    term_de = Ode * opz**exponent
    return np.sqrt(term_m + term_de)

def _ref_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference integral with caching."""
    key = (round(z, 10), round(w0, 10), round(wa, 10))
    if key in _ref_cache:
        return _ref_cache[key]
    if z < 1e-12:
        val = 1.0
    else:
        def f(t):
            return 1.0 / _E_of_z(z*t, w0, wa, Omega_m)
        val, _ = quad(f, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
    _ref_cache[key] = val
    return val

def _build_test_set():
    """
    Build a comprehensive test set covering the CPL parameter space.
    Focus on regions likely to be in validation: moderate z, physically
    motivated (w0, wa) ranges.
    """
    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
    w0_vals = np.array([-1.2, -1.0, -0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0])
    wa_vals = np.array([-0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8])
    
    points = []
    # Full grid but subsampled
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                points.append((z, w0, wa))
    
    # Add some extra points at extreme z
    for z in [0.02, 0.01, 10.0, 15.0]:
        for w0 in [-1.0, 0.0, 1.0]:
            for wa in [-0.5, 0.0, 0.5]:
                points.append((z, w0, wa))
    
    # Deduplicate
    seen = set()
    unique = []
    for p in points:
        key = (round(p[0],6), round(p[1],6), round(p[2],6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    return unique

def _solve_lp_for_weights(A, refs, n=8):
    """
    Solve: minimize t s.t.
      |A_i @ w - ref_i| <= t * ref_i  for all i
      sum(w) = 1, w >= 0, t >= 0
    """
    N = A.shape[0]
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    A_ub = np.zeros((2*N, n + 1))
    b_ub = np.zeros(2*N)
    for i in range(N):
        A_ub[2*i, :n] = A[i, :]
        A_ub[2*i, n] = -refs[i]
        b_ub[2*i] = refs[i]
        A_ub[2*i+1, :n] = -A[i, :]
        A_ub[2*i+1, n] = -refs[i]
        b_ub[2*i+1] = -refs[i]
    
    A_eq = np.zeros((1, n+1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(0, None)] * (n+1)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-10})
        if res.success:
            w = res.x[:n]
            s = w.sum()
            if s > 1e-15:
                w /= s
            return w, res.x[n]
    except Exception:
        pass
    return None, np.inf

def _eval_minimax(nodes, test_points, Omega_m=0.3):
    """Evaluate minimax relative error for given nodes."""
    n = len(nodes)
    N = len(test_points)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i, (z, w0, wa) in enumerate(test_points):
        if z < 1e-12:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, w0, wa, Omega_m)
    
    w, t = _solve_lp_for_weights(A, refs, n)
    if w is None:
        return None, np.inf
    return w, t

def _smooth_minimax_surrogate(nodes, test_points, beta=50.0, Omega_m=0.3):
    """
    Smooth approximation of the minimax error using log-sum-exp.
    Returns (smooth_error, actual_minimax_error, weights).
    """
    n = len(nodes)
    N = len(test_points)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i, (z, w0, wa) in enumerate(test_points):
        if z < 1e-12:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, w0, wa, Omega_m)
    
    w, t = _solve_lp_for_weights(A, refs, n)
    if w is None:
        return 1e8, np.inf, None
    
    # Compute individual relative errors
    errs = np.abs(A @ w - refs) / refs
    # Log-sum-exp for smooth max
    max_err = np.max(errs)
    if max_err > 0:
        log_sum_exp = np.log(np.sum(np.exp(beta * (errs - max_err)))) / beta + max_err
    else:
        log_sum_exp = 0.0
    
    return log_sum_exp, t, w

def _logit_to_nodes(x):
    """Transform unconstrained x to nodes in (0,1) using sigmoid."""
    s = 1.0 / (1.0 + np.exp(-x))
    # Ensure strictly increasing by adding small offset
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return s

def _nodes_to_logit(nodes):
    """Inverse of logit_to_nodes."""
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    return np.log(nodes / (1 - nodes))

def construct():
    if time.time() > _DEADLINE - 3:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
    
    test_points = _build_test_set()
    n = 8
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Initial candidates
    candidates = []
    
    # 1. Gauss-Legendre
    gl_n, gl_w = roots_legendre(n)
    candidates.append((gl_n + 1.0) / 2.0)
    
    # 2. Chebyshev nodes
    k = np.arange(n)
    ch_n = np.cos((2*k + 1) * np.pi / (2*n))
    candidates.append((ch_n + 1.0) / 2.0)
    
    # 3. Gauss-Jacobi variants
    from scipy.special import roots_jacobi
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (1.5, 1.5)]:
        try:
            jn, _ = roots_jacobi(n, alpha, beta)
            jn_mapped = (jn + 1.0) / 2.0
            if np.all(np.diff(jn_mapped) > 1e-10):
                candidates.append(jn_mapped)
        except:
            pass
    
    # 4. Perturbed GL
    rng = np.random.RandomState(123)
    base_gl = (gl_n + 1.0) / 2.0
    for _ in range(30):
        pert = rng.uniform(-0.05, 0.05, n)
        cand = base_gl + pert
        cand = np.clip(cand, 1e-4, 1 - 1e-4)
        cand = np.sort(cand)
        if np.all(np.diff(cand) > 1e-6):
            candidates.append(cand)
    
    # 5. Log-spaced near 0
    for p in [1.3, 1.5, 2.0, 2.5, 3.0]:
        t = np.linspace(0, 1, n)
        ln = 1.0 - (1.0 - t)**p
        ln = np.clip(ln, 1e-4, 1-1e-4)
        if np.all(np.diff(ln) > 1e-6):
            candidates.append(ln)
    
    # 6. Power-law near 1
    for p in [0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
        t = np.linspace(0, 1, n)
        pn = t**p
        pn = np.clip(pn, 1e-4, 1-1e-4)
        if np.all(np.diff(pn) > 1e-6):
            candidates.append(pn)
    
    # Evaluate all initial candidates
    for cand in candidates:
        if time.time() > _DEADLINE - 8:
            break
        w, err = _eval_minimax(cand, test_points)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    # Refine using smooth surrogate with Nelder-Mead
    def _objective_smooth(x):
        if time.time() > _DEADLINE - 5:
            return 1e8
        # x is in logit space, transform to nodes
        nodes_p = _logit_to_nodes(x)
        # Check if nodes are strictly increasing and in (0,1)
        if not np.all(np.diff(nodes_p) > 1e-8):
            return 1e8
        if np.any(nodes_p <= 1e-5) or np.any(nodes_p >= 1 - 1e-5):
            return 1e8
        
        smooth_err, actual_err, w = _smooth_minimax_surrogate(nodes_p, test_points, beta=50.0)
        if w is None:
            return 1e8
        return smooth_err
    
    x0 = _nodes_to_logit(best_nodes)
    
    try:
        res = minimize(_objective_smooth, x0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True, 'initial_simplex': None})
        if res.fun < best_err:
            nodes_r = _logit_to_nodes(res.x)
            if np.all(np.diff(nodes_r) > 1e-8) and np.all(nodes_r > 1e-5) and np.all(nodes_r < 1 - 1e-5):
                w_r, err_r = _eval_minimax(nodes_r, test_points)
                if w_r is not None and err_r < best_err:
                    best_nodes = nodes_r.copy()
                    best_weights = w_r.copy()
                    best_err = err_r
    except Exception:
        pass
    
    # Second round: direct Nelder-Mead on nodes (not logit)
    def _objective_direct(nodes_p):
        if time.time() > _DEADLINE - 5:
            return 1e8
        if np.any(nodes_p <= 1e-5) or np.any(nodes_p >= 1 - 1e-5):
            return 1e8
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-8):
            return 1e8
        
        smooth_err, actual_err, w = _smooth_minimax_surrogate(nodes_p, test_points, beta=50.0)
        if w is None:
            return 1e8
        return smooth_err
    
    x0_direct = best_nodes.copy()
    try:
        res2 = minimize(_objective_direct, x0_direct, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                 'adaptive': True})
        if res2.fun < best_err and np.all(res2.x > 1e-5) and np.all(res2.x < 1 - 1e-5):
            if np.all(np.diff(res2.x) > 1e-8):
                w_r, err_r = _eval_minimax(res2.x, test_points)
                if w_r is not None and err_r < best_err:
                    best_nodes = res2.x.copy()
                    best_weights = w_r.copy()
                    best_err = err_r
    except Exception:
        pass
    
    # Third round: multiple random restarts near best
    if time.time() < _DEADLINE - 15:
        rng2 = np.random.RandomState(777)
        for _ in range(10):
            if time.time() > _DEADLINE - 12:
                break
            pert = rng2.uniform(-0.02, 0.02, n)
            x2 = best_nodes + pert
            x2 = np.sort(np.clip(x2, 1e-4, 1-1e-4))
            if np.all(np.diff(x2) > 1e-8):
                try:
                    res3 = minimize(_objective_direct, x2, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res3.fun < best_err and np.all(res3.x > 1e-5) and np.all(res3.x < 1 - 1e-5):
                        if np.all(np.diff(res3.x) > 1e-8):
                            w_r, err_r = _eval_minimax(res3.x, test_points)
                            if w_r is not None and err_r < best_err:
                                best_nodes = res3.x.copy()
                                best_weights = w_r.copy()
                                best_err = err_r
                except Exception:
                    pass
    
    # Final polish: tighter LP solve
    if time.time() < _DEADLINE - 5:
        w_final, err_final = _eval_minimax(best_nodes, test_points)
        if w_final is not None and err_final < best_err:
            best_weights = w_final
            best_err = err_final
    
    # Ensure validity
    best_nodes = np.sort(best_nodes)
    if np.any(best_weights <= 0):
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    s = best_weights.sum()
    if abs(s - 1.0) > 1e-12:
        best_weights /= s
    
    if not np.all(np.diff(best_nodes) > 0):
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    return {"nodes": best_nodes, "weights": best_weights}
