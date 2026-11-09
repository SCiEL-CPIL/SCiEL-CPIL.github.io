"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from sympy import (symbols, Rational, Poly, diff, real_roots, pi as sympy_pi,
                   Matrix, laguerre as sympy_laguerre, expand)

# EVOLVE-BLOCK-START

# Pre-compute Laguerre polynomial coefficients for degrees 0, 2, ..., 50 with alpha = -1/2
# L_n^alpha(x) = sum_{k=0}^n binom(n+alpha, n-k) (-x)^k / k!
# We need exact rational coefficients.
_LAG_COEFFS = []
_LAG_DERIV_COEFFS = []

def _init_laguerre_coeffs():
    global _LAG_COEFFS, _LAG_DERIV_COEFFS
    if _LAG_COEFFS:
        return
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    for d in range(0, 51, 2):
        try:
            lp = sympy_laguerre(d, alpha, x)
            # Get coefficients as a list [c_0, c_1, ..., c_d] such that lp = sum c_k x^k
            coeff_list = [lp.coeff(x, k) for k in range(d + 1)]
            # Ensure all are Rational
            coeff_list = [Rational(c) for c in coeff_list]
            _LAG_COEFFS.append(coeff_list)
            
            # Derivative coefficients
            deriv_lp = lp.diff(x)
            deriv_coeff_list = [deriv_lp.coeff(x, k) for k in range(d)]
            deriv_coeff_list = [Rational(c) for c in deriv_coeff_list]
            _LAG_DERIV_COEFFS.append(deriv_coeff_list)
        except Exception:
            # Fallback if something goes wrong, though it shouldn't
            _LAG_COEFFS.append([0])
            _LAG_DERIV_COEFFS.append([0])

# Initialize on module load
_init_laguerre_coeffs()

def _horner_eval(coeffs, x_val):
    """Evaluate polynomial sum coeffs[i] * x^i at x_val using Horner's method."""
    result = Rational(0)
    for c in reversed(coeffs):
        result = result * x_val + c
    return result

def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic with fast eval.
    Returns None if invalid or timeout."""
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    
    # Build matrix A (26x26) and vector b (26x1)
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Precompute L_{2k}(0) and L'_{2k}(0)
    # L(0) is just the constant term coeffs[0]
    # L'(0) is the linear term coeffs[1] (if exists)
    for k in range(26):
        coeffs = _LAG_COEFFS[k]
        deriv_coeffs = _LAG_DERIV_COEFFS[k]
        
        # Value at 0
        A[0, k] = coeffs[0]
        
        # Derivative at 0
        if len(deriv_coeffs) > 0:
            A[1, k] = deriv_coeffs[0]
        else:
            A[1, k] = 0
            
    b[0, 0] = 0
    b[1, 0] = 1
    
    # For each z_i
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        
        for k in range(26):
            coeffs = _LAG_COEFFS[k]
            deriv_coeffs = _LAG_DERIV_COEFFS[k]
            
            A[row_val, k] = _horner_eval(coeffs, zi)
            A[row_der, k] = _horner_eval(deriv_coeffs, zi)
            
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
    # We can construct the polynomial g(x) by summing the coefficient lists
    g_coeffs = [Rational(0)] * 51 # Max degree is 50
    
    for k in range(26):
        c_k = c[k]
        if c_k == 0:
            continue
        coeffs = _LAG_COEFFS[k]
        for j in range(len(coeffs)):
            g_coeffs[j] += c_k * coeffs[j]
            
    # Build denominator: x * prod_i (x - z_i)^2
    # We need to perform exact polynomial division g(x) / denom(x)
    # It's easier to work with Poly objects for division
    g_poly = Poly(sum(g_coeffs[j] * x**j for j in range(51) if g_coeffs[j] != 0), x)
    
    # Construct denom poly
    denom_expr = x
    for zi in z_rationals:
        denom_expr = denom_expr * (x - zi)**2
    denom_poly = Poly(denom_expr, x)
    
    if time.time() > deadline:
        return None
    
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


def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 560.0  # 70s safety margin before 630s kill
    
    # Strategy: Use fast evaluator to allow more search iterations.
    # 1. Generate diverse initial candidates.
    # 2. Evaluate them.
    # 3. Perform local hill-climbing on the best one.
    
    candidates = []
    
    # Family 1: Bessel J0 zeros squared (baseline)
    z_bessel = []
    for k in range(1, 13):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            j0 = 2.4048255576957728
        else:
            j0 = x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
        z_bessel.append(j0**2)
    if _is_valid_z(z_bessel):
        candidates.append(z_bessel)
        
    # Family 2: Geometric progressions
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        for start in [0.5, 1.0, 2.0]:
            z_geo = [start * (ratio ** i) for i in range(12)]
            if _is_valid_z(z_geo):
                candidates.append(z_geo)

    # Family 3: Arithmetic progressions
    for start in [0.5, 1.0, 2.0]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            z_arith = [start + step * i for i in range(12)]
            if _is_valid_z(z_arith):
                candidates.append(z_arith)

    # Family 4: Random-ish (deterministic seed)
    import random
    random.seed(42)
    for _ in range(20):
        z_rand = [random.uniform(0.5, 50.0) for _ in range(12)]
        z_rand.sort()
        if _is_valid_z(z_rand):
            candidates.append(z_rand)

    # Evaluate all candidates and pick the best
    best_score = float('inf')
    best_z = z_bessel  # fallback
    
    # Sort candidates by some heuristic? No, just evaluate.
    # Limit evaluations to avoid timeout if candidates are too many
    max_evals = 50
    for idx, z_cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        if idx >= max_evals:
            break
        try:
            score = _compute_score_exact(z_cand, deadline - 10)
            if score is not None and score < best_score:
                best_score = score
                best_z = list(z_cand)
        except Exception:
            continue
    
    if time.time() > deadline - 60:
        return best_z
    
    # Local refinement: Hill climbing with adaptive steps
    current_z = list(best_z)
    current_score = best_score
    
    # Use a set of step sizes, decreasing over time
    base_steps = [0.1, 0.05, 0.01, 0.001]
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        # Try perturbing each coordinate
        for i in range(12):
            if time.time() > deadline - 30:
                break
            
            # Try multiple step sizes
            for step in base_steps:
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
                        score = _compute_score_exact(candidate, deadline - 10)
                        if score is not None and score < current_score:
                            current_score = score
                            current_z = candidate
                            improved = True
                    except Exception:
                        continue

    return current_z

# EVOLVE-BLOCK-END
