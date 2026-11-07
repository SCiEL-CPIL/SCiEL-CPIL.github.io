"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from scipy.optimize import differential_evolution, minimize, brentq


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
    
    Q_float = np.array(Q, dtype=np.float64)
    
    try:
        roots = np.roots(Q_float)
    except Exception:
        return None
    
    real_pos_roots = []
    for r in roots:
        if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real)):
            if r.real > 0:
                real_pos_roots.append(float(r.real))
                
    if not real_pos_roots:
        return None
        
    real_pos_roots.sort(reverse=True)
    
    def Qval(y):
        return np.polyval(Q_float[::-1], y)

    def _check_sign_change(y):
        for h in [1e-5, 1e-6, 1e-7, 1e-8]:
            h = h * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                return True
        return False

    def _refine_root(y):
        for h in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            h = h * max(1.0, abs(y))
            lo = max(0.0, y - h)
            hi = y + h
            try:
                flo = Qval(lo)
                fhi = Qval(hi)
                if flo * fhi < 0:
                    return brentq(Qval, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)
            except Exception:
                continue
        return y

    for y in real_pos_roots:
        if _check_sign_change(y):
            y_refined = _refine_root(y)
            if _check_sign_change(y_refined):
                return y_refined
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


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 210.0
    MARGIN = 25.0

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

    # Phase 1: Broad exploration with Differential Evolution
    bounds = [(-5.0, 5.0), (-0.5, 0.5)]
    
    try:
        if time_left() > 60:
            res_de = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=200,
                popsize=30,
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

    # Phase 2: Local refinement with Nelder-Mead
    # The objective is continuous in the region where the largest root is unique and simple.
    # Nelder-Mead is robust to the non-smoothness at root collisions (which are measure zero).
    try:
        if time_left() > 30:
            # Use a slightly wider initial simplex for robustness
            x0 = best_params
            res_nm = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'maxiter': 10000,
                    'maxfev': 10000,
                    'adaptive': True
                }
            )
            if res_nm is not None and np.isfinite(res_nm.fun):
                if res_nm.fun < best_val:
                    best_val = float(res_nm.fun)
                    best_params = np.array(res_nm.x, dtype=np.float64)
    except Exception:
        pass

    # Phase 3: High-precision polish with Powell (derivative-free, more robust than NM in some cases)
    try:
        if time_left() > 15:
            res_pw = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'xatol': 1e-14,
                    'fatol': 1e-14,
                    'maxiter': 1000,
                    'maxfev': 10000
                }
            )
            if res_pw is not None and np.isfinite(res_pw.fun):
                if res_pw.fun < best_val:
                    best_val = float(res_pw.fun)
                    best_params = np.array(res_pw.x, dtype=np.float64)
    except Exception:
        pass

    # Final check
    if time_left() < 0:
        return np.array([seed_c0, seed_c0 * seed_c1r, seed_c0 * seed_c2r])

    c1_opt = best_params[0]
    c2_opt = best_params[1]
    
    # Return [c0, c1, c2]
    # We can scale arbitrarily, but keeping c0 close to seed or normalized is fine.
    # The evaluator computes c3 based on the ratio, so scale doesn't matter.
    # Let's return the normalized version scaled back to c0 = -1.13 for consistency, 
    # or just the normalized [1, c1, c2] scaled by c0_seed.
    # Actually, the task says "Choose three finite coefficients c0, c1, c2".
    # The evaluator builds g = c0 H0 + ...
    # c3 is fixed by g(0)=0.
    # The value depends only on the shape of q(x), which is invariant under scaling of g.
    # So we can return [1, c1_opt, c2_opt] or [-1.13, ...].
    # Let's return the normalized coefficients to keep numbers small.
    return np.array([1.0, c1_opt, c2_opt])

# EVOLVE-BLOCK-END
