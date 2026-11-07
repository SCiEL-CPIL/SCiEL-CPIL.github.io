"""Optimize the uncertainty-inequality constant via continuous optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import differential_evolution, minimize, brentq, basinhopping


# ---- Precompute Hermite polynomial coefficients (physicists' convention) ----
def _hermite_coeffs(n):
    """Return list of Fractions for H_n(x) = sum a_k x^k (k=0..n)."""
    if n == 0:
        return [Fraction(1)]
    if n == 1:
        return [Fraction(0), Fraction(2)]
    c1 = _hermite_coeffs(n - 1)
    c2 = _hermite_coeffs(n - 2)
    a = [Fraction(0)] + [Fraction(2) * v for v in c1]
    b = [Fraction(-2 * (n - 1)) * v for v in c2]
    m = max(len(a), len(b))
    res = [Fraction(0)] * m
    for i, v in enumerate(a):
        res[i] += v
    for i, v in enumerate(b):
        res[i] += v
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res


_HC = {n: _hermite_coeffs(n) for n in (0, 4, 8, 12)}
_HF = {n: np.array([float(v) for v in c], dtype=np.float64) for n, c in _HC.items()}
_H12_0 = float(_HC[12][0])
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])


def _build_Q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return the float coefficient array of Q(y) = q(sqrt(y)),
    where q(x) = g(x)/x^2.
    Q(y) is a polynomial in y of degree 5.
    Coefficients are ordered [Q_0, Q_1, ..., Q_5].
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    # g(x) = 1*H0 + c1*H4 + c2*H8 + c3*H12
    # g(x) = sum_{k=0}^6 g_{2k} x^{2k}
    # Q(y) = sum_{k=0}^6 g_{2k} y^k
    # We only need coefficients for x^2, x^4, ..., x^12
    # _HF[n] is [a0, a1, ..., an]
    # We extract even indices.
    
    g_coeffs = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g_coeffs[2] is coeff of x^2, g_coeffs[4] of x^4, etc.
    # Q(y) = g_coeffs[2] + g_coeffs[4]y + ... + g_coeffs[12]y^5
    Q = g_coeffs[2:13:2]
    # Trim trailing zeros
    while len(Q) > 1 and Q[-1] == 0.0:
        Q = Q[:-1]
    return Q


def _largest_sign_change_root_Q(Q):
    """
    Find the largest positive real root y of Q(y) at which Q changes sign.
    Q is a numpy array of coefficients ascending.
    Returns the root y (float) or None.
    """
    if len(Q) < 2:
        return None
    try:
        roots = np.roots(Q)
    except Exception:
        return None
    
    real_pos_roots = []
    for r in roots:
        if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
            if r.real > 0:
                real_pos_roots.append(float(r.real))
                
    if not real_pos_roots:
        return None
        
    real_pos_roots.sort(reverse=True)
    
    def Qval(y):
        v = 0.0
        for c in Q[::-1]:
            v = v * y + c
        return v

    for y in real_pos_roots:
        h = 1e-6 * max(1.0, abs(y))
        left = Qval(y - h)
        right = Qval(y + h)
        if left == 0.0 or right == 0.0:
            h2 = 1e-9 * max(1.0, abs(y))
            left = Qval(y - h2)
            right = Qval(y + h2)
        if left * right < 0:
            # Refine
            try:
                lo = y - h
                hi = y + h
                flo = Qval(lo)
                fhi = Qval(hi)
                if flo * fhi < 0:
                    y_refined = brentq(Qval, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
                    return y_refined
            except Exception:
                pass
            return y
    return None


def _value_from(c1, c2):
    """Compute the objective value = r^2/(2*pi) = y_max/(2*pi)."""
    try:
        Q = _build_Q_coeffs(c1, c2)
        y_max = _largest_sign_change_root_Q(Q)
        if y_max is None:
            return np.inf
        return y_max / (2.0 * np.pi)
    except Exception:
        return np.inf


def _objective(params):
    """Vectorized-safe objective for scipy (params = [c1, c2])."""
    c1, c2 = params
    return _value_from(float(c1), float(c2))

def _objective_3d(params):
    """3D objective for scipy (params = [c0, c1, c2])."""
    c0, c1, c2 = params
    if abs(c0) < 1e-10:
        return np.inf
    c1r = c1 / c0
    c2r = c2 / c0
    return _value_from(float(c1r), float(c2r))

def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 200.0
    MARGIN = 10.0

    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    
    # Normalize to c0 = 1
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0

    best_params = np.array([seed_c1r, seed_c2r])
    best_val = _value_from(seed_c1r, seed_c2r)

    def time_left():
        return DEADLINE - MARGIN - (time.time() - start)

    # Phase 1: 3D Differential Evolution to explore the full space
    # This handles the scaling invariance explicitly and might find better basins
    try:
        if time_left() > 80:
            # Bounds for c0, c1, c2
            # c0 should not be 0. Let's bound it away from 0.
            # c1, c2 can be large.
            bounds_3d = [
                (-5.0, 5.0),  # c0
                (-5.0, 5.0),  # c1
                (-1.0, 1.0)   # c2
            ]
            
            res_de_3d = differential_evolution(
                _objective_3d,
                bounds=bounds_3d,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.8,
                seed=42,
                polish=True,
                init='sobol',
                workers=-1,
            )
            if res_de_3d is not None and np.isfinite(res_de_3d.fun):
                if res_de_3d.fun < best_val:
                    best_val = float(res_de_3d.fun)
                    c0_opt, c1_opt, c2_opt = res_de_3d.x
                    # Normalize back to c0=1 for 2D refinement
                    if abs(c0_opt) > 1e-10:
                        best_params = np.array([c1_opt / c0_opt, c2_opt / c0_opt], dtype=np.float64)
    except Exception:
        pass

    # Phase 2: 2D Differential Evolution (refinement in normalized space)
    try:
        if time_left() > 60:
            # Wider bounds to allow exploration
            bounds = [(-5.0, 5.0), (-0.5, 0.5)]
            
            res_de = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=150,
                popsize=20,
                tol=1e-10,
                mutation=(0.5, 1.5),
                recombination=0.8,
                seed=42,
                polish=True,
                init='sobol',
                workers=-1,
            )
            if res_de is not None and np.isfinite(res_de.fun):
                if res_de.fun < best_val:
                    best_val = float(res_de.fun)
                    best_params = np.array(res_de.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 3: Basin Hopping
    try:
        if time_left() > 30:
            def _basin_obj(x):
                return _objective(x)
            
            res_bh = basinhopping(
                _basin_obj,
                x0=best_params,
                niter=20,
                stepsize=0.5,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'xatol': 1e-10, 'fatol': 1e-12}},
                seed=456,
            )
            if res_bh is not None and np.isfinite(res_bh.fun):
                if res_bh.fun < best_val:
                    best_val = float(res_bh.fun)
                    best_params = np.array(res_bh.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 4: Local Polish (Nelder-Mead)
    try:
        if time_left() > 10:
            res_l = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14},
            )
            if res_l is not None and np.isfinite(res_l.fun):
                if res_l.fun < best_val:
                    best_val = float(res_l.fun)
                    best_params = np.array(res_l.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 5: Final BFGS Polish
    try:
        if time_left() > 5:
            res_b = minimize(
                _objective,
                best_params,
                method='BFGS',
                options={'maxiter': 5000, 'gtol': 1e-12},
            )
            if res_b is not None and np.isfinite(res_b.fun):
                if res_b.fun < best_val:
                    best_val = float(res_b.fun)
                    best_params = np.array(res_b.x, dtype=np.float64)
    except Exception:
        pass

    # Convert back to (c0, c1, c2)
    # We can return any valid scaling. c0=1 is safe.
    c0_out = 1.0
    c1_out = float(best_params[0])
    c2_out = float(best_params[1])

    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])

    return np.array([c0_out, c1_out, c2_out])
# EVOLVE-BLOCK-END
