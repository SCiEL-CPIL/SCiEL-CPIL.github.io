"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import laguerre as _laguerre_gen
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # Use sympy's built-in laguerre function for exactness
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(_laguerre_gen(d, alpha, x))
    
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
        # g(zi) = 0
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        # g'(zi) = 0
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
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    # 1e-198 as rational
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

def construct():
    """Return 12 double-root locations z_1..z_12 using roots of L_12^{-1/2}(x)."""
    from sympy import symbols, Rational
    from sympy.polys.orthopolys import laguerre
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Compute the 12th generalized Laguerre polynomial
    L12 = laguerre(12, alpha, x)
    
    # Find the roots numerically to get floats
    # nroots returns complex numbers, we filter for real positive
    roots = L12.nroots()
    z_list = []
    for r in roots:
        re, im = r.as_real_imag()
        if abs(im) < 1e-10:
            val = float(re)
            if val > 0:
                z_list.append(val)
                
    # Ensure we have 12 distinct positive roots
    if len(z_list) != 12:
        # Fallback to small linear spacing if something goes wrong
        z_list = [i * 0.01 for i in range(1, 13)]
    else:
        # Sort them
        z_list.sort()
        
    return z_list

# EVOLVE-BLOCK-END
