"""
Strategy: Minimax quadrature optimization with structured test set coverage.

We seek 8 nodes and 8 positive weights summing to 1 that minimize the worst-case
relative error across a broad range of CPL cosmological parameters (w0, wa) 
and redshifts z.

The integrand is f(t) = 1/E(z*t) where E(z) = H(z)/H0.
E^2 = Om*(1+z)^3 + Ode*(1+z)^(3+3w0+3wa*z)

Approach:
1. Construct a structured test set covering the parameter space boundaries 
   and interior, as extreme parameter combinations often yield the worst errors.
2. For a given set of nodes, solve a Linear Program (LP) to find weights that 
   minimize the maximum relative error against high-precision reference integrals.
3. Optimize the node positions using Nelder-Mead on the resulting minimax error.
4. Use multiple initial candidates (Gauss-Legendre, Chebyshev, Jacobi, Power-law) 
   and restarts to avoid local minima.
5. Final polish: Re-solve for weights using a denser structured test set.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Vectorized E(z) = H(z)/H0 for CPL model."""
    Ode = 1.0 - Omega_m
    opz = 1.0 + z
    term_m = Omega_m * opz**3
    exponent = 3.0 + 3.0*w0 + 3.0*wa*z
    term_de = Ode * opz**exponent
    return np.sqrt(term_m + term_de)

def _ref_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference integral of 1/E(z*t) dt from 0 to 1."""
    if z < 1e-12:
        return 1.0
    def f(t):
        return 1.0 / _E_of_z(z*t, w0, wa, Omega_m)
    val, _ = quad(f, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
    return val

def _build_test_set(dense=False):
    """
    Build a structured test set covering the CPL parameter space.
    Focuses on boundaries and axes where errors are typically maximized.
    """
    # Z values: logarithmic spacing
    if dense:
        z_vals = np.logspace(np.log10(0.01), np.log10(10.0), 25)
    else:
        z_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    
    # W0, Wa ranges
    if dense:
        w0_vals = np.linspace(-1.5, 1.5, 15)
        wa_vals = np.linspace(-1.0, 1.0, 15)
    else:
        w0_vals = np.array([-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
        wa_vals = np.array([-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0])
    
    points = []
    # 1. Cover axes: wa=0 for all w0, w0=0 for all wa
    for z in z_vals:
        for w0 in w0_vals:
            points.append((z, w0, 0.0))
        for wa in wa_vals:
            if wa != 0.0:
                points.append((z, 0.0, wa))
    
    # 2. Cover corners and interior grid points
    # Subsample the full grid to keep LP tractable but ensure coverage
    if dense:
        # Full grid for dense set is 25 * 15 * 15 = 5625, too many for fast LP in loop? 
        # Actually, 5625 * 2 constraints = 11250 rows. HiGHS is fast, but let's subsample slightly if needed.
        # For now, let's take a stratified sample of the interior if it gets too big.
        pass
        
    # Add a set of interior points to catch non-axis maxima
    # We'll add a subset of the full grid
    n_interior = 100 if not dense else 300
    rng = np.random.RandomState(42)
    
    # Generate random interior points
    for _ in range(n_interior):
        z = 10**rng.uniform(np.log10(0.01), np.log10(10.0))
        w0 = rng.uniform(-1.5, 1.5)
        wa = rng.uniform(-1.0, 1.0)
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
    
    # Vectorized construction of A_ub and b_ub is faster
    # Row 2i:   A[i, :] @ w - t * refs[i] <= refs[i]
    # Row 2i+1: -A[i, :] @ w - t * refs[i] <= -refs[i]
    
    A_ub[0::2, :n] = A
    A_ub[1::2, :n] = -A
    
    A_ub[0::2, n] = -refs
    A_ub[1::2, n] = -refs
    
    b_ub[0::2] = refs
    b_ub[1::2] = -refs
    
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

def _eval_minimax(nodes, test_points, Omega_m=0.3):
    """Evaluate minimax relative error for given nodes."""
    n = len(nodes)
    N = len(test_points)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    
    # Pre-compute references (this is the bottleneck, but necessary for accuracy)
    # In a real speedup, we might cache these, but nodes change so A changes.
    # Refs only depend on (z, w0, wa), not nodes.
    # We should cache refs outside this function if possible, but for simplicity 
    # and to avoid memory bloat with multiple test sets, we compute here.
    # Optimization: Compute refs only once per unique (z,w0,wa) in the test set.
    # Since test_points is fixed during optimization, we can pre-compute refs.
    
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

def construct():
    if time.time() > _DEADLINE - 3:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
    
    n = 8
    
    # Use a moderate structured test set for optimization
    # Pre-compute references for this set to speed up _eval_minimax? 
    # The current _eval_minimax computes refs every time. 
    # Let's create a wrapper that caches refs for the specific test set used in optimization.
    
    opt_test_points = _build_test_set(dense=False)
    
    # Cache references for opt_test_points
    opt_refs = np.zeros(len(opt_test_points))
    for i, (z, w0, wa) in enumerate(opt_test_points):
        opt_refs[i] = _ref_integral(z, w0, wa)
        
    # Helper to evaluate using cached refs
    def _eval_minimax_cached(nodes):
        N = len(opt_test_points)
        A = np.zeros((N, n))
        for i, (z, w0, wa) in enumerate(opt_test_points):
            if z < 1e-12:
                A[i, :] = 1.0
            else:
                zt = z * nodes
                E = _E_of_z(zt, w0, wa)
                A[i, :] = 1.0 / E
        w, t = _solve_lp_for_weights(A, opt_refs, n)
        if w is None:
            return None, np.inf
        return w, t

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
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (1.5, 1.5)]:
        try:
            jn, _ = roots_jacobi(n, alpha, beta)
            jn_mapped = (jn + 1.0) / 2.0
            if np.all(np.diff(jn_mapped) > 1e-10):
                candidates.append(jn_mapped)
        except:
            pass
            
    # 4. Random perturbations of GL
    rng = np.random.RandomState(123)
    base_gl = (gl_n + 1.0) / 2.0
    for _ in range(15):
        pert = rng.uniform(-0.05, 0.05, n)
        cand = base_gl + pert
        cand = np.clip(cand, 1e-4, 1 - 1e-4)
        cand = np.sort(cand)
        if np.all(np.diff(cand) > 1e-6):
            candidates.append(cand)

    # 5. Power law nodes
    for p in [0.5, 0.7, 1.0, 1.5, 2.0]:
        t = np.linspace(0, 1, n)
        pn = t**p
        pn = np.clip(pn, 1e-4, 1-1e-4)
        if np.all(np.diff(pn) > 1e-6):
            candidates.append(pn)
            
    # 6. Log-spaced nodes
    for p in [1.3, 1.5, 2.0]:
        t = np.linspace(0, 1, n)
        ln = 1.0 - (1.0 - t)**p
        ln = np.clip(ln, 1e-4, 1-1e-4)
        if np.all(np.diff(ln) > 1e-6):
            candidates.append(ln)

    # Evaluate all initial candidates
    for cand in candidates:
        if time.time() > _DEADLINE - 15:
            break
        w, err = _eval_minimax_cached(cand)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    # Refinement with Nelder-Mead
    def _objective(x):
        nodes_p = x.copy()
        if np.any(nodes_p <= 1e-4) or np.any(nodes_p >= 1-1e-4):
            return 1e8
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-7):
            return 1e8
        w, err = _eval_minimax_cached(nodes_p)
        if w is None:
            return 1e8
        return err
    
    # Run Nelder-Mead from the best candidate
    try:
        res = minimize(_objective, best_nodes, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        if res.fun < best_err and np.all(res.x > 1e-4) and np.all(res.x < 1-1e-4):
            if np.all(np.diff(res.x) > 1e-7):
                w_r, err_r = _eval_minimax_cached(res.x)
                if w_r is not None and err_r < best_err:
                    best_nodes = res.x.copy()
                    best_weights = w_r.copy()
                    best_err = err_r
    except Exception:
        pass
        
    # Second round: Multiple restarts around the best solution
    if time.time() < _DEADLINE - 10:
        rng2 = np.random.RandomState(777)
        for _ in range(5):
            if time.time() > _DEADLINE - 8:
                break
            pert = rng2.normal(0, 0.02, n)
            start = best_nodes + pert
            start = np.clip(start, 1e-4, 1-1e-4)
            start = np.sort(start)
            if not np.all(np.diff(start) > 1e-7):
                continue
            try:
                res2 = minimize(_objective, start, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res2.fun < best_err and np.all(res2.x > 1e-4) and np.all(res2.x < 1-1e-4):
                    if np.all(np.diff(res2.x) > 1e-7):
                        w_r, err_r = _eval_minimax_cached(res2.x)
                        if w_r is not None and err_r < best_err:
                            best_nodes = res2.x.copy()
                            best_weights = w_r.copy()
                            best_err = err_r
            except Exception:
                pass

    # Final high-precision polish pass
    if time.time() < _DEADLINE - 5:
        # Use a larger, denser test set for the final weight solve
        fine_test_points = _build_test_set(dense=True)
        fine_refs = np.zeros(len(fine_test_points))
        for i, (z, w0, wa) in enumerate(fine_test_points):
            fine_refs[i] = _ref_integral(z, w0, wa)
            
        N = len(fine_test_points)
        A = np.zeros((N, n))
        for i, (z, w0, wa) in enumerate(fine_test_points):
            if z < 1e-12:
                A[i, :] = 1.0
            else:
                zt = z * best_nodes
                E = _E_of_z(zt, w0, wa)
                A[i, :] = 1.0 / E
                
        w_fine, err_fine = _solve_lp_for_weights(A, fine_refs, n)
        if w_fine is not None:
            best_weights = w_fine
            best_err = err_fine
    
    # Final cleanup
    best_nodes = np.sort(best_nodes)
    
    # Ensure validity
    if np.any(best_weights <= 0):
        # Fallback: use Gauss-Legendre
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    s = best_weights.sum()
    if abs(s - 1.0) > 1e-12:
        best_weights /= s
    
    if not np.all(np.diff(best_nodes) > 0):
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    return {"nodes": best_nodes, "weights": best_weights}
