"""Optimize the uncertainty-inequality constant via smooth surrogate and multi-start optimization."""
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


def _build_q_coeffs(c1, c2):
    """
    Given c1, c2 (with c0=1), return float coefficient array of q(x) = g(x)/x^2.
    q(x) is degree 10: q(x) = sum_{k=0}^{5} g_{2(k+1)} x^{2k}
    Coefficients ascending: [q_0, q_1, ..., q_5] where q_i is coeff of x^{2i}.
    """
    c3 = -(_H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g_coeffs = 1.0 * _HF[0] + c1 * _HF[4] + c2 * _HF[8] + c3 * _HF[12]
    # g_coeffs[2] is coeff of x^2, g_coeffs[4] of x^4, etc.
    q = g_coeffs[2:13:2]
    while len(q) > 1 and q[-1] == 0.0:
        q = q[:-1]
    return q


def _value_from(c1, c2):
    """Compute r^2/(2*pi). Returns inf if no valid sign-change root."""
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return np.inf
        
        # Find roots of q(x) = 0. q is in x^{2k} so let y = x^2, Q(y) = q(sqrt(y))
        # Q(y) = q_0 + q_1*y + ... + q_5*y^5
        Q = q  # same coefficients, just variable change
        
        try:
            roots_y = np.roots(Q[::-1])  # np.roots expects descending order
        except Exception:
            return np.inf
        
        # Filter: real, positive roots
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)):
                if r.real > 1e-12:
                    real_pos.append(float(r.real))
        
        if not real_pos:
            return 1e10
        
        def Qval(y):
            return np.polyval(Q[::-1], y)
        
        # Check sign change at each root
        best_y = None
        for y in sorted(real_pos, reverse=True):
            h = 1e-7 * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                best_y = y
                break
        
        if best_y is None:
            # Try with larger h
            for y in sorted(real_pos, reverse=True):
                for h in [1e-6, 1e-5, 1e-4, 1e-3]:
                    h = h * max(1.0, abs(y))
                    left = Qval(y - h)
                    right = Qval(y + h)
                    if left * right < 0:
                        best_y = y
                        break
                if best_y is not None:
                    break
        
        if best_y is None:
            return 1e10
        
        return best_y / (2.0 * np.pi)
    except Exception:
        return np.inf


def _value_smooth(c1, c2, beta=500.0):
    """
    Smooth surrogate for _value_from using soft-max over sign-change roots.
    Instead of taking the largest sign-change root, use a soft-max (log-sum-exp)
    which is differentiable and allows gradient-based optimization.
    """
    try:
        q = _build_q_coeffs(c1, c2)
        if len(q) < 2:
            return 1e10
        
        Q = q
        
        try:
            roots_y = np.roots(Q[::-1])
        except Exception:
            return 1e10
        
        # Filter: real, positive roots
        real_pos = []
        for r in roots_y:
            if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real)):
                if r.real > 1e-10:
                    real_pos.append(float(r.real))
        
        if not real_pos:
            return 1e10
        
        def Qval(y):
            return np.polyval(Q[::-1], y)
        
        # Find sign-change roots
        sign_change_roots = []
        for y in real_pos:
            h = 1e-6 * max(1.0, abs(y))
            left = Qval(y - h)
            right = Qval(y + h)
            if left * right < 0:
                sign_change_roots.append(y)
        
        if not sign_change_roots:
            return 1e10
        
        # Soft-max: log(sum(exp(beta * y))) / beta approximates max(y)
        # Use log-sum-exp for numerical stability
        y_arr = np.array(sign_change_roots)
        max_y = np.max(y_arr)
        if max_y < 0:
            return 1e10
        log_sum = max_y / beta + np.log(np.sum(np.exp(beta * (y_arr - max_y) / beta)))
        soft_max_y = log_sum
        
        return soft_max_y / (2.0 * np.pi)
    except Exception:
        return 1e10


def construct():
    """Return three coefficients [c0, c1, c2] minimizing r^2/(2*pi)."""
    start = time.time()
    DEADLINE = 200.0  # 10s margin before 210s kill
    
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
    
    # Multi-start optimization using smooth surrogate
    from scipy.optimize import minimize
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    start_points = [(seed_c1r, seed_c2r)]
    
    # Add perturbations around seed
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(4):
            c1_p = seed_c1r + rng.normal(0, scale)
            c2_p = seed_c2r + rng.normal(0, scale * 0.1)
            start_points.append((c1_p, c2_p))
    
    # Add some hand-tuned points
    hand_points = [
        (0.0, 0.0),
        (0.5, 0.05),
        (-0.5, 0.05),
        (1.0, 0.1),
        (-1.0, 0.1),
        (0.3, 0.03),
        (-0.3, 0.03),
        (0.8, 0.08),
        (-0.8, 0.08),
        (0.1, 0.01),
        (-0.1, 0.01),
        (0.6, 0.06),
        (-0.6, 0.06),
    ]
    start_points.extend(hand_points)
    
    # Evaluate all start points with exact value
    for c1_s, c2_s in start_points:
        if time_left() < 15:
            break
        v = _value_from(c1_s, c2_s)
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_c1 = c1_s
            best_c2 = c2_s
    
    # Run Nelder-Mead from best point and several other promising starts
    def objective(params):
        return _value_smooth(params[0], params[1], beta=300.0)
    
    # Try multiple local optimizations
    local_starts = [
        (best_c1, best_c2),
        (seed_c1r, seed_c2r),
        (seed_c1r + 0.2, seed_c2r + 0.02),
        (seed_c1r - 0.2, seed_c2r + 0.02),
        (seed_c1r + 0.2, seed_c2r - 0.02),
        (seed_c1r - 0.2, seed_c2r - 0.02),
        (0.0, 0.0),
        (0.5, 0.05),
        (-0.5, 0.05),
    ]
    
    for c1_s, c2_s in local_starts:
        if time_left() < 20:
            break
        try:
            res = minimize(
                objective,
                [c1_s, c2_s],
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if np.isfinite(res.fun):
                # Verify with exact value
                v_exact = _value_from(res.x[0], res.x[1])
                if np.isfinite(v_exact) and v_exact < best_val:
                    best_val = v_exact
                    best_c1 = float(res.x[0])
                    best_c2 = float(res.x[1])
        except Exception:
            pass
    
    # Final polish with tighter tolerances on the best point
    if time_left() > 10:
        def objective_tight(params):
            return _value_smooth(params[0], params[1], beta=1000.0)
        
        try:
            res = minimize(
                objective_tight,
                [best_c1, best_c2],
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
            )
            if np.isfinite(res.fun):
                v_exact = _value_from(res.x[0], res.x[1])
                if np.isfinite(v_exact) and v_exact < best_val:
                    best_val = v_exact
                    best_c1 = float(res.x[0])
                    best_c2 = float(res.x[1])
        except Exception:
            pass
    
    # One more COBYLA polish
    if time_left() > 5:
        try:
            res = minimize(
                lambda p: _value_from(p[0], p[1]),
                [best_c1, best_c2],
                method='COBYLA',
                options={'maxiter': 3000, 'rhobeg': 1e-9, 'tol': 1e-14},
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
