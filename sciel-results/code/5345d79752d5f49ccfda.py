"""
Strategy:
Use the roots of L_12^(-1/2)(x) as initial double-root locations, then perform
a coordinate-descent local search to minimize B (the largest real root of q with
sign change). Each step evaluates B exactly using sympy rational arithmetic.
"""
import time
import math
import sympy
from sympy import Rational, symbols, expand, diff, Matrix, LUsolve, genlaguerre, div

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Uses Laguerre roots as initialization, then local search to minimize B.
    """
    t0 = time.time()
    DEADLINE = 600
    SAFETY_MARGIN = 60  # generous safety margin for exact arithmetic
    
    def elapsed():
        return time.time() - t0
    
    def time_ok():
        return elapsed() < DEADLINE - SAFETY_MARGIN
    
    try:
        # Compute roots of L_12^(-1/2)(x) as initial z
        x_sym = symbols('x')
        alpha = Rational(-1, 2)
        L_poly = genlaguerre(12, alpha, x_sym)
        roots = L_poly.roots()
        z_list = sorted([float(r) for r in roots])
        
        if len(z_list) != 12 or not all(zi > 0 for zi in z_list) or len(set(z_list)) != 12:
            z_list = [4.0 * (i + 1) for i in range(12)]
        
        # Evaluate initial B
        initial_B = _evaluate_B(z_list, t0, DEADLINE, SAFETY_MARGIN)
        if initial_B is None:
            return z_list
        
        best_z = list(z_list)
        best_B = initial_B
        
        # Coordinate descent: try perturbing each z_i
        # Use small relative perturbations
        max_rounds = 3  # limited by time
        for round_idx in range(max_rounds):
            if not time_ok():
                break
            improved = False
            for i in range(12):
                if not time_ok():
                    break
                # Try several perturbations for z_i
                for factor in [0.9, 0.95, 1.05, 1.1]:
                    if not time_ok():
                        break
                    new_z = list(best_z)
                    new_z[i] = best_z[i] * factor
                    # Validate
                    if any(zi <= 0 for zi in new_z):
                        continue
                    if len(set(new_z)) != 12:
                        continue
                    B_val = _evaluate_B(new_z, t0, DEADLINE, SAFETY_MARGIN)
                    if B_val is not None and B_val < best_B:
                        best_B = B_val
                        best_z = new_z
                        improved = True
                        break  # move to next coordinate
        return best_z
        
    except Exception:
        pass
    
    # Fallback
    return [4.0 * (i + 1) for i in range(12)]


def _evaluate_B(z_list, t0, DEADLINE, SAFETY_MARGIN):
    """
    Evaluate B for a given list of 12 z values.
    Returns B (float) or None if invalid/timeout.
    """
    def time_ok():
        return time.time() - t0 < DEADLINE - SAFETY_MARGIN
    
    if not time_ok():
        return None
    
    try:
        # Convert z to rationals
        z_rat = [Rational(float(zi)) for zi in z_list]
        
        # Check distinctness and positivity
        if len(set(z_rat)) != 12:
            return None
        if any(zi <= 0 for zi in z_rat):
            return None
        
        x = symbols('x')
        alpha = Rational(-1, 2)
        
        # Build Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
        laguerres = []
        for k in range(26):
            d = 2 * k
            L = genlaguerre(d, alpha, x)
            laguerres.append(L)
        
        if not time_ok():
            return None
        
        # Build the system: g(x) = sum c_k * L_{2k}(x)
        # Constraints: g(0)=0, g'(0)=1, g(z_i)=g'(z_i)=0 for all i
        c_syms = [symbols(f'c_{i}') for i in range(26)]
        
        g = sum(c_syms[k] * laguerres[k] for k in range(26))
        g_diff = diff(g, x)
        
        # Build matrix A and vector b: A * c = b
        A = Matrix(26, 26, 0)
        b = Matrix(26, 1, 0)
        
        # Row 0: g(0) = 0
        g0 = g.subs(x, 0)
        for col, c_sym in enumerate(c_syms):
            A[0, col] = g0.coeff(c_sym)
        b[0] = -g0.subs(c_syms, [0]*26)
        
        # Row 1: g'(0) = 1
        g1 = g_diff.subs(x, 0)
        for col, c_sym in enumerate(c_syms):
            A[1, col] = g1.coeff(c_sym)
        b[1] = 1 - g1.subs(c_syms, [0]*26)
        
        # Rows 2..25: g(z_i)=0 and g'(z_i)=0
        for idx, zi in enumerate(z_rat):
            row_g = 2 + 2 * idx
            row_gp = 3 + 2 * idx
            
            gz = g.subs(x, zi)
            for col, c_sym in enumerate(c_syms):
                A[row_g, col] = gz.coeff(c_sym)
            b[row_g] = -gz.subs(c_syms, [0]*26)
            
            gzi = g_diff.subs(x, zi)
            for col, c_sym in enumerate(c_syms):
                A[row_gp, col] = gzi.coeff(c_sym)
            b[row_gp] = -gzi.subs(c_syms, [0]*26)
        
        if not time_ok():
            return None
        
        # Check singularity
        det_val = A.det()
        if det_val == 0:
            return None
        
        # Solve
        c_sol = A.LUsolve(b)
        
        if not time_ok():
            return None
        
        # Build g with solved coefficients
        g_sol = sum(c_sol[k] * laguerres[k] for k in range(26))
        g_sol = expand(g_sol)
        
        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rat:
            denom = denom * (x - zi)**2
        denom = expand(denom)
        
        # Polynomial division: q = g / denom
        q, remainder = div(g_sol, denom, x)
        
        if not time_ok():
            return None
        
        # Find real roots of q
        real_roots = sympy.real_roots(q)
        
        if not time_ok():
            return None
        
        # Count roots with sign change at r +/- 1e-198
        # 1e-198 as rational
        eps = Rational(1, 10**198)
        
        B = 0
        for r in real_roots:
            if not time_ok():
                break
            r_val = r
            # Evaluate q at r - eps and r + eps
            q_minus = q.subs(x, r_val - eps)
            q_plus = q.subs(x, r_val + eps)
            
            # Check if strictly opposite signs
            if q_minus != 0 and q_plus != 0:
                sign_minus = 1 if q_minus > 0 else -1
                sign_plus = 1 if q_plus > 0 else -1
                if sign_minus != sign_plus:
                    r_float = float(r_val)
                    if r_float > B:
                        B = r_float
        
        return B
    
    except Exception:
        return None
