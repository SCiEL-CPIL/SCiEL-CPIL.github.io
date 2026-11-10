"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

# Precompute Laguerre polynomials L_{2k}^{(-1/2)}(x) as sympy expressions
# Do this once at module level to avoid recomputation
_LAGUERRE_POLS = None
_LAGUERRE_DIFFS = None
_x_sym = None
_alpha_sym = None

def _init_laguerre():
    """Initialize and cache the Laguerre polynomials."""
    global _LAGUERRE_POLS, _LAGUERRE_DIFFS, _x_sym, _alpha_sym
    if _LAGUERRE_POLS is not None:
        return
    from sympy import symbols, Rational, laguerre
    _x_sym = symbols('x')
    _alpha_sym = Rational(-1, 2)
    _LAGUERRE_POLS = []
    _LAGUERRE_DIFFS = []
    for d in range(0, 51, 2):
        lp = laguerre(d, _alpha_sym, _x_sym)
        _LAGUERRE_POLS.append(lp)
        _LAGUERRE_DIFFS.append(lp.diff(_x_sym))

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic.
    Returns None if invalid or timeout."""
    from sympy import (Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix)
    
    if time.time() > deadline:
        return None
    
    _init_laguerre()
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    x = _x_sym
    
    if time.time() > deadline:
        return None
    
    # Build matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    # Row 1: g'(0) = 1
    for k in range(26):
        A[0, k] = _LAGUERRE_POLS[k].subs(x, 0)
        A[1, k] = _LAGUERRE_DIFFS[k].subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    # For each z_i
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = _LAGUERRE_POLS[k].subs(x, zi)
            A[row_der, k] = _LAGUERRE_DIFFS[k].subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    # Solve the linear system
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Build g(x) = sum c_k * L_{2k}(x)
    g = sum(c[k] * _LAGUERRE_POLS[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)


def _is_valid_z(z_list):
    """Check if z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-12:
                return False
    return True


def _bessel_zero_J0_approx(k):
    """Approximate the k-th zero of Bessel J0."""
    if k == 1:
        return 2.4048255576957728
    x = (k - 0.25) * math.pi
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)


def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 580.0  # 50s safety margin before 630s kill
    
    # Strategy: Try multiple initial configurations, pick the best, then refine
    # Key insight: We want B (largest real root of q) to be as small as possible.
    
    candidates = []
    
    # Family 1: Bessel J0 zeros squared (baseline)
    z_bessel = [_bessel_zero_J0_approx(k)**2 for k in range(1, 13)]
    if _is_valid_z(z_bessel):
        candidates.append(('bessel', z_bessel))
    
    # Family 2: Squares of integers
    z_squares = [float(i**2) for i in range(1, 13)]
    if _is_valid_z(z_squares):
        candidates.append(('squares', z_squares))
    
    # Family 3: (2i-1)^2/4 pattern
    z_half = [float((2*i-1)**2)/4.0 for i in range(1, 13)]
    if _is_valid_z(z_half):
        candidates.append(('half', z_half))
    
    # Family 4: Geometric progression
    for ratio in [1.5, 2.0, 2.5]:
        z_geo = [0.5 * (ratio ** i) for i in range(12)]
        if _is_valid_z(z_geo):
            candidates.append((f'geo_{ratio}', z_geo))
    
    # Family 5: Linear spacing with various starts/steps
    for start in [0.5, 1.0, 2.0]:
        for step in [1.0, 2.0, 3.0]:
            z_lin = [start + step * i for i in range(12)]
            if _is_valid_z(z_lin):
                candidates.append((f'lin_{start}_{step}', z_lin))
    
    # Family 6: Interpolate between Bessel and squares
    for t in [0.2, 0.4, 0.6, 0.8]:
        z_interp = [(1-t)*z_bessel[i] + t*z_squares[i] for i in range(12)]
        if _is_valid_z(z_interp):
            candidates.append((f'interp_{t}', z_interp))
    
    # Evaluate all candidates and pick the best
    best_score = float('inf')
    best_z = z_bessel  # fallback
    
    for name, z_cand in candidates:
        if time.time() > deadline - 60:
            break
        try:
            score = _compute_score_exact(z_cand, deadline - 20)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(z_cand)
        except Exception:
            continue
    
    if time.time() > deadline - 30:
        return best_z
    
    # Local refinement: perturb each z_i and see if score improves
    current_z = list(best_z)
    current_score = best_score
    
    # Adaptive step sizes - start larger, then smaller
    step_sizes = [0.1, 0.05, 0.02, 0.01]
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for i in range(12):
            if time.time() > deadline - 30:
                break
            for step in step_sizes:
                for sign in [1, -1]:
                    if time.time() > deadline - 20:
                        break
                    candidate = list(current_z)
                    candidate[i] = current_z[i] * (1 + sign * step)
                    if candidate[i] <= 0:
                        continue
                    if not _is_valid_z(candidate):
                        continue
                    try:
                        score = _compute_score_exact(candidate, deadline - 20)
                        if score is not None and score < current_score:
                            current_score = score
                            current_z = candidate
                            improved = True
                    except Exception:
                        continue
    
    return current_z

# EVOLVE-BLOCK-END
