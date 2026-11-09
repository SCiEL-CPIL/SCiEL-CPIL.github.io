"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, real_roots, pi as sympy_pi, Matrix, binomial
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / math.factorial(k)
        return result
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    # Build matrix A and vector b
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
    
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
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
    
    # Check if division is exact (remainder should be 0)
    if q_poly is None:
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
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 570.0  # Safety margin before 630s kill
    
    # Strategy: Try multiple initializations and pick the best
    # Based on observation, small clustered values may work well
    
    # Initialization 1: Small values clustered near 0
    z_init1 = [0.1 * (i + 1) for i in range(12)]
    
    # Initialization 2: Values related to Laguerre zeros (approximate)
    # Zeros of L_n^(alpha) for small n
    z_init2 = [0.5 * (i + 1)**1.5 for i in range(12)]
    
    # Initialization 3: Evenly spaced small values
    z_init3 = [i * 0.3 + 0.1 for i in range(12)]
    
    # Initialization 4: Geometric spacing
    z_init4 = [0.05 * (2.0 ** i) for i in range(12)]
    
    # Initialization 5: Very small values
    z_init5 = [0.01 * (i + 1) for i in range(12)]
    
    candidates = [z_init1, z_init2, z_init3, z_init4, z_init5]
    
    best_score = float('inf')
    best_result = None
    
    for z_init in candidates:
        if _time_remaining(deadline) < 90:
            break
        
        # Validate
        valid = True
        for i in range(12):
            if not math.isfinite(z_init[i]) or z_init[i] <= 0:
                valid = False
                break
        if valid:
            for i in range(12):
                for j in range(i+1, 12):
                    if abs(z_init[i] - z_init[j]) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            continue
        
        try:
            score = _compute_score_exact(z_init)
            if score is not None and score < best_score:
                best_score = score
                best_result = list(z_init)
        except Exception:
            pass
    
    # If we have a good result, try to refine with local search
    if best_result is not None and _time_remaining(deadline) > 120:
        # Try small perturbations of each coordinate
        for i in range(12):
            if _time_remaining(deadline) < 90:
                break
            for delta_frac in [0.05, -0.05, 0.1, -0.1, 0.02, -0.02]:
                if _time_remaining(deadline) < 90:
                    break
                candidate = list(best_result)
                candidate[i] = best_result[i] * (1 + delta_frac)
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
                        best_result = candidate
                except Exception:
                    pass
    
    # Ensure we always return a valid result
    if best_result is None:
        # Fallback: use small positive distinct values
        best_result = [0.1 * (i + 1) for i in range(12)]
    
    return best_result
# EVOLVE-BLOCK-END

