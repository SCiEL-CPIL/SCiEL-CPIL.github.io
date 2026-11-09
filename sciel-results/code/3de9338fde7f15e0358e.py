"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import genlaguerre
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build the 26 generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(genlaguerre(d, alpha, x))
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Precompute L_{2k}(0) and L'_{2k}(0)
    L0_vals = [poly.subs(x, 0) for poly in laguerre_pols]
    L0_deriv_vals = [poly.diff(x).subs(x, 0) for poly in laguerre_pols]
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = L0_vals[k]
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = L0_deriv_vals[k]
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
    
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None  # singular system
    
    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
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
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
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

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use a logarithmic spacing that places roots more densely near 0.
    This is based on the intuition that for Laguerre-based interpolation with
    alpha = -1/2, the natural scale is related to the parameter, and roots
    near 0 may help control the behavior of q(x) to push the largest valid
    real root B as low as possible.
    
    We try a geometric progression: z_i = a * r^(i-1) for i=1..12,
    and search over a few (a, r) pairs.
    """
    deadline = time.time() + 570.0  # 630s kill, keep 60s margin for safety
    
    # Candidate configurations: (a, r) where z_i = a * r^(i-1)
    # We want all z_i > 0 and distinct.
    # Try several geometric spacings.
    
    candidates = []
    
    # Geometric progressions with various ratios
    for a in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        for r in [1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
            z_list = [a * (r ** (i)) for i in range(12)]
            # Check validity
            valid = True
            for z in z_list:
                if not math.isfinite(z) or z <= 0:
                    valid = False
                    break
            if valid:
                for i in range(12):
                    for j in range(i+1, 12):
                        if abs(z_list[i] - z_list[j]) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                candidates.append(z_list)
    
    # Also try some arithmetic progressions
    for a in [0.5, 1.0, 2.0]:
        for d in [1.0, 2.0, 3.0, 5.0]:
            z_list = [a + d * i for i in range(12)]
            valid = True
            for z in z_list:
                if not math.isfinite(z) or z <= 0:
                    valid = False
                    break
            if valid:
                candidates.append(z_list)
    
    # Also try the Bessel-based approach as a fallback
    z_bessel = [((i + 0.75) * math.pi - 1/(8*(i+0.75)*math.pi))**2 for i in range(12)]
    # Actually use proper asymptotic for J0 zeros
    def bessel_zero_J0(k):
        # j_{0,k} ≈ (k - 1/4)*π - 1/(8*(k-1/4)*π) + 9/(128*(k-1/4)^3*π^3)
        x = (k - 0.25) * math.pi
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3)
    
    z_bessel = [bessel_zero_J0(i+1)**2 for i in range(12)]
    candidates.append(z_bessel)
    
    # Evaluate candidates, keeping track of the best
    best_score = float('inf')
    best_result = candidates[0] if candidates else [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    for cand in candidates:
        if _time_remaining(deadline) < 90:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_score:
                best_score = score
                best_result = list(cand)
        except Exception:
            continue
    
    # If we have time, try local refinement around the best result
    if _time_remaining(deadline) > 120 and best_result is not None:
        result = list(best_result)
        for i in range(12):
            if _time_remaining(deadline) < 90:
                break
            # Try small multiplicative perturbations
            for delta_frac in [0.05, -0.05, 0.1, -0.1]:
                if _time_remaining(deadline) < 90:
                    break
                candidate = list(result)
                candidate[i] = result[i] * (1 + delta_frac)
                if candidate[i] <= 0:
                    continue
                # Check distinctness
                valid = True
                for a in range(12):
                    for b_idx in range(a+1, 12):
                        if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                try:
                    score = _compute_score_exact(candidate)
                    if score is not None and score < best_score:
                        best_score = score
                        result = candidate
                except Exception:
                    continue
    
    return result
# EVOLVE-BLOCK-END
