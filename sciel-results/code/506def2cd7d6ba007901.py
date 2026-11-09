"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)


def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic. Returns None on failure."""
    try:
        from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, LUsolve
        from sympy.polys.orthopolys import laguerre as _laguerre_gen
        
        z_rationals = sorted(Rational(float(z)) for z in z_list)
        
        x = symbols('x')
        alpha = Rational(-1, 2)
        
        # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
        # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        # We need to solve for c_k from:
        #   g(0) = 0
        #   g'(0) = 1
        #   g(z_i) = 0 and g'(z_i) = 0 for each i=1..12
        # That's 2 + 12*2 = 26 equations for 26 unknowns c_0..c_25
        
        # Precompute Laguerre polynomials L_{2k}^{alpha}(x) for k=0..25
        L_poly = []
        for k in range(26):
            d = 2 * k
            L_d = _laguerre_gen(d, alpha, x)
            L_poly.append(L_d)
        
        # Build the linear system
        # Variables: c_0, c_1, ..., c_25
        # Equations:
        #   eq0: g(0) = sum_k c_k * L_{2k}(0) = 0
        #   eq1: g'(0) = sum_k c_k * L'_{2k}(0) = 1
        #   eq_{2i}: g(z_i) = sum_k c_k * L_{2k}(z_i) = 0
        #   eq_{2i+1}: g'(z_i) = sum_k c_k * L'_{2k}(z_i) = 0
        
        A = Matrix(26, 26, lambda r, c: 0)
        b_vec = Matrix(26, 1, lambda r, c: 0)
        
        # Evaluate L_{2k}(0) and L'_{2k}(0)
        for k in range(26):
            L_val = L_poly[k].subs(x, 0)
            L_deriv_val = diff(L_poly[k], x).subs(x, 0)
            A[0, k] = L_val
            A[1, k] = L_deriv_val
        
        b_vec[0] = 0
        b_vec[1] = 1
        
        # For each z_i
        for i in range(12):
            z_i = z_rationals[i]
            row_g = 2 + 2 * i
            row_gp = 2 + 2 * i + 1
            
            for k in range(26):
                L_val = L_poly[k].subs(x, z_i)
                L_deriv_val = diff(L_poly[k], x).subs(x, z_i)
                A[row_g, k] = L_val
                A[row_gp, k] = L_deriv_val
            
            b_vec[row_g] = 0
            b_vec[row_gp] = 0
        
        # Solve the system
        try:
            c_vec = A.LUsolve(b_vec)
        except Exception:
            return None
        
        # Build g(x) = sum_k c_k * L_{2k}(x)
        g = sum(c_vec[k] * L_poly[k] for k in range(26))
        
        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for i in range(12):
            denom = denom * (x - z_rationals[i])**2
        
        # q(x) = g(x) / denom (exact polynomial division)
        g_poly = Poly(g, x)
        denom_poly = Poly(denom, x)
        
        try:
            q_poly, rem = g_poly.div(denom_poly)
            if rem != 0:
                return None
        except Exception:
            return None
        
        # Find real roots of q
        try:
            roots = real_roots(q_poly.as_expr(), x)
        except Exception:
            return None
        
        # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
        B = 0
        for r in roots:
            r_val = float(r)
            if not math.isfinite(r_val):
                continue
            # Evaluate q at r - 1e-198 and r + 1e-198
            # Use rational arithmetic: 1e-198 is not a rational, so we need to be careful
            # The task says "q has strictly opposite signs at r - 1e-198 and r + 1e-198"
            # Since 1e-198 is not representable exactly, we interpret this as:
            # Check if q changes sign in a neighborhood of r
            # Use a small rational offset
            eps = Rational(1, 10**198)
            q_minus = q_poly.as_expr().subs(x, r - eps)
            q_plus = q_poly.as_expr().subs(x, r + eps)
            
            # Check if they have opposite signs
            q_minus_sign = 1 if q_minus > 0 else (-1 if q_minus < 0 else 0)
            q_plus_sign = 1 if q_plus > 0 else (-1 if q_plus < 0 else 0)
            
            if q_minus_sign != 0 and q_plus_sign != 0 and q_minus_sign != q_plus_sign:
                if r_val > B:
                    B = r_val
        
        return B / (2 * math.pi)
    except Exception:
        return None


def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 570.0  # 630s kill, keep 60s margin for safety
    
    # Bessel zeros of J_0: j_{0,k} ≈ (k - 1/4)π for large k
    # For our purpose, z_i = (j_{0,i})^2 captures the Laguerre zero distribution for α=-1/2
    
    # Initial guess: z_i = (j_{0,i})^2
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Ensure positive, finite, distinct
    result = []
    for z in z_bessel:
        if not math.isfinite(z) or z <= 0:
            result = None
            break
        result.append(z)
    
    if result is None or len(result) != 12:
        # Fallback
        result = [4.0 * (i + 1) for i in range(12)]
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    # Try to improve by testing slight variations if time permits
    # Test a few scale factors around 1.0
    best_score = None
    best_z = result
    
    # Evaluate the baseline
    if time.time() < deadline - 120:  # Need at least 120s for one full evaluation
        baseline_score = _compute_score_exact(result)
        if baseline_score is not None:
            best_score = baseline_score
    
    # Try a few variations: scale the z values
    scales_to_try = [0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    for scale in scales_to_try:
        if time.time() >= deadline - 120:
            break
        
        z_scaled = [z * scale for z in z_bessel]
        
        # Validate
        valid = True
        for z in z_scaled:
            if not math.isfinite(z) or z <= 0:
                valid = False
                break
        if not valid:
            continue
        
        # Check distinct
        for i in range(12):
            for j in range(i+1, 12):
                if abs(z_scaled[i] - z_scaled[j]) < 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        score = _compute_score_exact(z_scaled)
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
                best_z = z_scaled
    
    # Return the best found
    return best_z
# EVOLVE-BLOCK-END
