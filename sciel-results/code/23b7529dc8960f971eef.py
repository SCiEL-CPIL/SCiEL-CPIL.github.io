"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _bessel_zero_J1(k):
    """Approximate k-th zero of J_1 using asymptotic expansion."""
    x = (k + 0.25) * math.pi
    if x < 0.1:
        return 3.8317059702075125  # j_{1,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, laguerre
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(laguerre(d, x, alpha))
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
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

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z_list(z_list):
    """Check that z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def _generate_candidates():
    """Generate diverse candidate z-locations for optimization."""
    candidates = []
    
    # 1. J_0 zeros squared
    z_j0 = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j0):
        candidates.append(z_j0)
    
    # 2. J_1 zeros squared
    z_j1 = [_bessel_zero_J1(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j1):
        candidates.append(z_j1)
    
    # 3. Square roots of J_0 zeros (compressed)
    z_sqrt_j0 = [math.sqrt(_bessel_zero_J0(i)) for i in range(1, 13)]
    if _validate_z_list(z_sqrt_j0):
        candidates.append(z_sqrt_j0)
    
    # 4. Geometric progression with different ratios
    for ratio in [1.3, 1.5, 2.0]:
        z_geo = [1.0 * ratio**i for i in range(12)]
        if _validate_z_list(z_geo):
            candidates.append(z_geo)
    
    # 5. Arithmetic progressions
    for start, step in [(1.0, 2.0), (2.0, 3.0), (0.5, 1.5)]:
        z_arith = [start + step*i for i in range(12)]
        if _validate_z_list(z_arith):
            candidates.append(z_arith)
    
    # 6. Power law: z_i = a * i^p
    for a, p in [(1.0, 0.8), (2.0, 0.7), (1.5, 0.9)]:
        z_power = [a * (i+1)**p for i in range(12)]
        if _validate_z_list(z_power):
            candidates.append(z_power)
    
    # 7. Exponential spacing
    for base in [1.2, 1.4, 1.6]:
        z_exp = [math.exp(base * i / 11.0) for i in range(12)]
        if _validate_z_list(z_exp):
            candidates.append(z_exp)
    
    # Default fallback
    default_z = [4.0 * (i+1) for i in range(12)]
    if not candidates:
        candidates.append(default_z)
    
    return candidates

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    # 630s kill. Set deadline to 580s to allow for cleanup and return.
    deadline = time.time() + 580.0
    
    # Generate multiple candidate initial configurations
    candidates = _generate_candidates()
    
    best_score = float('inf')
    best_z = candidates[0]
    
    # Evaluate candidates in order of expected quality
    # Bessel zeros are most likely to be good
    eval_order = [0, 1, 2]  # J0^2, J1^2, sqrt(J0)
    for idx in eval_order:
        if _time_remaining(deadline) < 180:
            break
        if idx >= len(candidates):
            continue
        try:
            score = _compute_score_exact(candidates[idx])
            if score is not None and score < best_score:
                best_score = score
                best_z = candidates[idx]
        except Exception:
            pass
    
    # If we still have time, try a few more diverse candidates
    remaining_indices = [i for i in range(len(candidates)) if i not in eval_order]
    for idx in remaining_indices[:3]:
        if _time_remaining(deadline) < 150:
            break
        try:
            score = _compute_score_exact(candidates[idx])
            if score is not None and score < best_score:
                best_score = score
                best_z = candidates[idx]
        except Exception:
            pass
    
    # Final validation
    if not _validate_z_list(best_z):
        best_z = [4.0 * (i+1) for i in range(12)]
    
    return best_z
# EVOLVE-BLOCK-END
