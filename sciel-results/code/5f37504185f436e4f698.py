"""
Strategy: Differential Evolution node optimization for minimax quadrature.

Structural change: Replace the multi-phase (active-set + Nelder-Mead) approach
with a single global optimization using differential evolution (DE) on the
node positions. DE is better suited for the non-convex, non-smooth minimax
landscape because it maintains population diversity and avoids local minima
that trap gradient-free local methods.

Key structural differences from parent:
1. Single global optimizer (DE) instead of sequential local refinements.
2. Larger, more diverse training set sampled uniformly from the CPL box.
3. Direct evaluation of the LP-based minimax objective (no surrogate).
4. Logit parameterization to handle node ordering constraints naturally.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, linprog
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

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
    """High-accuracy reference integral of 1/E(z*t) dt from 0 to 1 (cached)."""
    if z < 1e-12:
        return 1.0
    key = (round(z, 10), round(w0, 10), round(wa, 10), round(Omega_m, 10))
    if key in _ref_cache:
        return _ref_cache[key]
    def f(t):
        return 1.0 / _E_of_z(z*t, w0, wa, Omega_m)
    val, _ = quad(f, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
    _ref_cache[key] = val
    return val

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

def _build_test_set():
    """
    Build a diverse test set: stratified sampling from the CPL parameter box
    plus targeted high-z and extreme-parameter points.
    """
    rng = np.random.RandomState(2024)
    
    # Uniform stratified sampling in (z, w0, wa)
    z_samples = np.sort(rng.uniform(0.01, 12.0, 200))
    w0_samples = rng.uniform(-1.5, 1.5, 200)
    wa_samples = rng.uniform(-1.0, 1.0, 200)
    
    points = list(zip(z_samples, w0_samples, wa_samples))
    
    # Add targeted points: high redshift
    for z in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for w0 in [-1.0, 0.0, 1.0]:
            for wa in [-1.0, 0.0, 1.0]:
                points.append((z, w0, wa))
    
    # Add Lambda-CDM (w0=-1, wa=0) at many z
    for z in np.logspace(-2, 1.1, 15):
        points.append((z, -1.0, 0.0))
    
    # Add phantom matter (w0=1, wa=0)
    for z in np.logspace(-2, 1.1, 10):
        points.append((z, 1.0, 0.0))
    
    # Deduplicate
    seen = set()
    unique = []
    for p in points:
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    # Cap at 350 for speed
    if len(unique) > 350:
        idx = rng.choice(len(unique), 350, replace=False)
        unique = [unique[i] for i in idx]
    
    return unique

def _eval_minimax(nodes, test_points, Omega_m=0.3):
    """Evaluate minimax relative error for given nodes. Returns (weights, max_rel_err)."""
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

def _nodes_to_logit(nodes):
    """Convert nodes in (0,1) to logit space."""
    return np.log(nodes / (1.0 - nodes))

def _logit_to_nodes(logit_vals):
    """Convert logit values back to (0,1) and sort."""
    nodes = 1.0 / (1.0 + np.exp(-logit_vals))
    return np.sort(nodes)

def _de_objective(logit_params, test_points, Omega_m=0.3):
    """
    Objective for differential evolution: returns the minimax LP error.
    Parameters are in logit space; we convert to nodes, solve LP, return max rel err.
    """
    if time.time() > _DEADLINE - 5:
        return 1e8
    
    nodes = _logit_to_nodes(logit_params)
    n = len(nodes)
    
    # Check validity
    if np.any(nodes <= 1e-6) or np.any(nodes >= 1 - 1e-6):
        return 1e8
    if not np.all(np.diff(nodes) > 1e-7):
        return 1e8
    
    w, err = _eval_minimax(nodes, test_points, Omega_m)
    if w is None:
        return 1e8
    return err

def construct():
    if time.time() > _DEADLINE - 3:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
    
    test_points = _build_test_set()
    n = 8
    
    # Build initial candidate pool
    candidates = []
    
    # Gauss-Legendre
    gl_n, gl_w = roots_legendre(n)
    candidates.append((gl_n + 1.0) / 2.0)
    
    # Chebyshev
    k = np.arange(n)
    ch_n = np.cos((2*k + 1) * np.pi / (2*n))
    candidates.append((ch_n + 1.0) / 2.0)
    
    # Gauss-Jacobi variants
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0)]:
        try:
            jn, _ = roots_jacobi(n, alpha, beta)
            jn_mapped = (jn + 1.0) / 2.0
            if np.all(np.diff(jn_mapped) > 1e-10):
                candidates.append(jn_mapped)
        except:
            pass
    
    # Perturbed GL
    rng = np.random.RandomState(123)
    base_gl = (gl_n + 1.0) / 2.0
    for _ in range(15):
        pert = rng.uniform(-0.05, 0.05, n)
        cand = base_gl + pert
        cand = np.clip(cand, 1e-4, 1 - 1e-4)
        cand = np.sort(cand)
        if np.all(np.diff(cand) > 1e-6):
            candidates.append(cand)
    
    # Power-law spacings
    for p in [0.7, 1.0, 1.3, 1.5, 2.0]:
        t = np.linspace(0, 1, n)
        pn = t**p
        pn = np.clip(pn, 1e-4, 1-1e-4)
        if np.all(np.diff(pn) > 1e-6):
            candidates.append(pn)
    
    # Evaluate initial candidates to find best starting point
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    for cand in candidates:
        if time.time() > _DEADLINE - 20:
            break
        w, err = _eval_minimax(cand, test_points)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    # Differential Evolution optimization in logit space
    if time.time() < _DEADLINE - 10:
        # Convert best initial nodes to logit space
        x0_logit = _nodes_to_logit(best_nodes)
        
        # Define bounds in logit space (corresponding to nodes in [0.001, 0.999])
        logit_bounds = [(np.log(0.001/0.999), np.log(0.999/0.001))] * n
        
        # Run differential evolution
        try:
            de_result = differential_evolution(
                _de_objective,
                bounds=logit_bounds,
                args=(test_points, 0.3),
                maxiter=60,
                popsize=15,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.8,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if de_result.success or de_result.fun < best_err:
                de_nodes = _logit_to_nodes(de_result.x)
                if np.all(np.diff(de_nodes) > 1e-7) and np.all(de_nodes > 1e-5) and np.all(de_nodes < 1-1e-5):
                    w_de, err_de = _eval_minimax(de_nodes, test_points)
                    if w_de is not None and err_de < best_err:
                        best_nodes = de_nodes.copy()
                        best_weights = w_de.copy()
                        best_err = err_de
        except Exception:
            pass
    
    # Second DE run with different seed if time permits
    if time.time() < _DEADLINE - 8:
        try:
            de_result2 = differential_evolution(
                _de_objective,
                bounds=[(np.log(0.001/0.999), np.log(0.999/0.001))] * n,
                args=(test_points, 0.3),
                maxiter=40,
                popsize=12,
                tol=1e-6,
                mutation=(0.6, 1.2),
                recombination=0.7,
                seed=99,
                polish=False,
                init='latinhypercube'
            )
            
            if de_result2.success or de_result2.fun < best_err:
                de_nodes2 = _logit_to_nodes(de_result2.x)
                if np.all(np.diff(de_nodes2) > 1e-7) and np.all(de_nodes2 > 1e-5) and np.all(de_nodes2 < 1-1e-5):
                    w_de2, err_de2 = _eval_minimax(de_nodes2, test_points)
                    if w_de2 is not None and err_de2 < best_err:
                        best_nodes = de_nodes2.copy()
                        best_weights = w_de2.copy()
                        best_err = err_de2
        except Exception:
            pass
    
    # Final polish: local refinement around best nodes using coordinate descent
    if time.time() < _DEADLINE - 5:
        rng_local = np.random.RandomState(2024)
        for _ in range(30):
            if time.time() > _DEADLINE - 4:
                break
            scale = 0.01 * (1.0 - _ / 30.0) + 0.001
            pert = rng_local.uniform(-scale, scale, n)
            cand = best_nodes + pert
            cand = np.sort(np.clip(cand, 1e-5, 1-1e-5))
            if not np.all(np.diff(cand) > 1e-7):
                continue
            w_l, err_l = _eval_minimax(cand, test_points)
            if w_l is not None and err_l < best_err:
                best_nodes = cand.copy()
                best_weights = w_l.copy()
                best_err = err_l
    
    # Final re-solve LP on a slightly different test set for robustness
    if time.time() < _DEADLINE - 3:
        rng_final = np.random.RandomState(777)
        z_f = rng_final.uniform(0.01, 10.0, 150)
        w0_f = rng_final.uniform(-1.5, 1.5, 150)
        wa_f = rng_final.uniform(-1.0, 1.0, 150)
        final_pts = list(zip(z_f, w0_f, wa_f))
        w_final, t_final = _eval_minimax(best_nodes, final_pts)
        if w_final is not None:
            best_weights = w_final
    
    # Ensure validity
    if best_weights is None or len(best_weights) != n:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
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
