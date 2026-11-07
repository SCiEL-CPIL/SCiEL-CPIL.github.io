"""Optimize the uncertainty-inequality constant via model-and-optimize approach."""
import time
import numpy as np
from fractions import Fraction


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
_H0_0 = float(_HC[0][0])
_H4_0 = float(_HC[4][0])
_H8_0 = float(_HC[8][0])
_H12_0 = float(_HC[12][0])


def _build_Q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return float coefficient array of Q(y) where
    Q(y) = g(x)/x^2 with y = x^2. Q is degree 5 in y.
    Returns coefficients [q_0, q_1, ..., q_5] for Q(y) = sum q_i * y^i.
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g_coeffs = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g_coeffs[2] is coeff of x^2, g_coeffs[4] of x^4, etc.
    # Q(y) has coeffs from g_coeffs[2], g_coeffs[4], ..., g_coeffs[12]
    Q = g_coeffs[2:13:2].copy()
    while len(Q) > 1 and Q[-1] == 0.0:
        Q = Q[:-1]
    return Q


def _largest_sign_change_root_Q(Q):
    """
    Find the largest positive real root of Q(y) at which Q changes sign.
    Returns y_max or None if no such root exists.
    """
    if len(Q) < 2:
        return None
    
    # Find all roots of Q(y) = 0
    try:
        roots_y = np.roots(Q[::-1])  # np.roots expects descending order
    except Exception:
        return None
    
    # Filter: real, positive roots
    real_pos = []
    for r in roots_y:
        if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
            if r.real > 1e-10:
                real_pos.append(float(r.real))
    
    if not real_pos:
        return None
    
    def Qval(y):
        return float(np.polyval(Q[::-1], y))
    
    # Check sign change at each root, starting from largest
    best_y = None
    for y in sorted(real_pos, reverse=True):
        # Use multiple step sizes to robustly detect sign change
        h = 1e-6 * max(1.0, abs(y))
        left = Qval(y - h)
        right = Qval(y + h)
        if left * right < 0:
            best_y = y
            break
        # Try larger steps if initial check fails
        for h_mult in [1e-5, 1e-4, 1e-3, 1e-2]:
            h = h_mult * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                best_y = y
                break
        if best_y is not None:
            break
    
    return best_y


def _value_from(c1, c2):
    """Compute r^2/(2*pi). Returns inf if no valid sign-change root."""
    try:
        Q = _build_Q_coeffs(c1, c2)
        y_max = _largest_sign_change_root_Q(Q)
        if y_max is None:
            return 1e10
        return y_max / (2.0 * np.pi)
    except Exception:
        return np.inf


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 205.0  # 5s margin
    
    seed_c0 = -113.0 / 100.0
    seed_c1 = 1.0 / 25.0
    seed_c2 = 1.0 / 3240.0
    
    # Normalize to c0 = 1
    seed_c1r = seed_c1 / seed_c0
    seed_c2r = seed_c2 / seed_c0
    
    best_c1 = seed_c1r
    best_c2 = seed_c2r
    best_val = _value_from(best_c1, best_c2)
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    # Phase 1: Coarse grid search to find promising regions
    if time_left() > 30:
        c1_range = np.linspace(-2.0, 2.0, 50)
        c2_range = np.linspace(-0.2, 0.2, 50)
        
        for c1 in c1_range:
            if time_left() < 25:
                break
            for c2 in c2_range:
                if time_left() < 20:
                    break
                v = _value_from(c1, c2)
                if v < best_val:
                    best_val = v
                    best_c1 = c1
                    best_c2 = c2
    
    # Phase 2: Local refinement using Nelder-Mead
    if time_left() > 20:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    # Phase 3: Try multiple starting points for Nelder-Mead
    if time_left() > 15:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        # Generate several candidate starting points
        candidates = [
            (best_c1, best_c2),
            (best_c1 + 0.1, best_c2),
            (best_c1 - 0.1, best_c2),
            (best_c1, best_c2 + 0.01),
            (best_c1, best_c2 - 0.01),
            (0.0, 0.0),
            (0.5, 0.05),
            (-0.5, 0.05),
            (1.0, 0.1),
            (-1.0, 0.1),
            (0.3, 0.03),
            (-0.3, 0.03),
            (0.8, 0.08),
            (-0.8, 0.08),
        ]
        
        for c1_init, c2_init in candidates:
            if time_left() < 10:
                break
            try:
                res = minimize(
                    objective,
                    [c1_init, c2_init],
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = float(res.fun)
                    best_c1, best_c2 = res.x
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time_left() > 5:
        from scipy.optimize import minimize
        
        def objective(params):
            return _value_from(params[0], params[1])
        
        try:
            res = minimize(
                objective,
                [best_c1, best_c2],
                method='COBYLA',
                options={'maxiter': 5000, 'rhobeg': 1e-10, 'tol': 1e-14},
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = float(res.fun)
                best_c1, best_c2 = res.x
        except Exception:
            pass
    
    if not np.isfinite(best_val):
        return np.array([seed_c0, seed_c1, seed_c2])
    
    c0_out = 1.0
    c1_out = float(best_c1)
    c2_out = float(best_c2)
    
    return np.array([c0_out, c1_out, c2_out])
