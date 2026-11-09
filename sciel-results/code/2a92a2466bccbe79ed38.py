import math
import time
import numpy as np
import sympy as sp
from scipy.optimize import minimize

# Global deadline
DEADLINE = time.time() + 590  # 590 seconds to be safe before 600s limit
START_TIME = time.time()

def _is_time_out():
    return time.time() > DEADLINE

def evaluate_B(z_list):
    """
    Evaluates the score B for a given list of 12 float z locations.
    Returns a large number if invalid or timeout.
    """
    if _is_time_out():
        return 1e100
    
    # Check validity
    if len(z_list) != 12:
        return 1e100
    for z in z_list:
        if not np.isfinite(z) or z <= 0:
            return 1e100
    
    # Check distinctness (with small tolerance)
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-5:
                return 1e100

    # Convert to exact Rationals
    # Note: float to Rational conversion is exact for the float value
    z_rationals = [sp.Rational(float(z)) for z in z_list]
    
    # Sort z_rationals for consistent polynomial construction? 
    # The problem says "list is sorted" for the final output, but the system 
    # g(z_i)=0 is symmetric with respect to permutation of z_i?
    # g(x) is defined by the set of double roots. The linear system is symmetric.
    # The polynomial q(x) depends on the set.
    # So order doesn't matter for the value of B.
    
    x = sp.Symbol('x')
    
    # Laguerre polynomials L_2k^(alpha)(x) with alpha = -1/2
    alpha = sp.Rational(-1, 2)
    
    # We need to solve for c_0 ... c_25
    # g(x) = sum_{k=0}^{25} c_k L_{2k}(x)
    # Conditions:
    # 1. g(0) = 0
    # 2. g'(0) = 1
    # 3. g(z_i) = 0 for i=1..12
    # 4. g'(z_i) = 0 for i=1..12
    
    # Total 26 equations for 26 unknowns.
    
    # Construct Laguerre polynomials L_0, L_2, ..., L_50
    # sympy.special.laguerre_poly(n, alpha, x)
    Ls = [sp.laguerre_poly(2*k, alpha, x) for k in range(26)]
    
    # Construct the linear system
    # Matrix A (26x26), Vector b (26)
    A = sp.zeros(26, 26)
    b = sp.zeros(26, 1)
    
    # Eq 1: g(0) = 0
    # sum c_k L_{2k}(0) = 0
    for k in range(26):
        A[0, k] = Ls[k].subs(x, 0)
    b[0] = 0
    
    # Eq 2: g'(0) = 1
    # sum c_k L'_{2k}(0) = 1
    for k in range(26):
        A[1, k] = sp.diff(Ls[k], x).subs(x, 0)
    b[1] = 1
    
    # Eqs 3-26: g(z_i)=0 and g'(z_i)=0
    idx = 2
    for i in range(12):
        zi = z_rationals[i]
        # g(zi) = 0
        for k in range(26):
            A[idx, k] = Ls[k].subs(x, zi)
        b[idx] = 0
        idx += 1
        
        # g'(zi) = 0
        for k in range(26):
            A[idx, k] = sp.diff(Ls[k], x).subs(x, zi)
        b[idx] = 0
        idx += 1
        
        if _is_time_out():
            return 1e100

    # Solve the linear system
    try:
        # LU solve is faster and more stable for this size
        c_vec = A.LUsolve(b)
    except (sp.matrices.exceptions.SingularMatrixError, ZeroDivisionError, Exception):
        return 1e100
        
    if _is_time_out():
        return 1e100

    # Construct g(x)
    g_poly = sum(c_vec[k] * Ls[k] for k in range(26))
    g_poly = sp.expand(g_poly)
    
    # Construct denominator d(x) = x * prod (x - z_i)^2
    d_poly = x
    for zi in z_rationals:
        d_poly = d_poly * (x - zi)**2
    
    # Polynomial division q(x) = g(x) / d(x)
    # d(x) degree is 1 + 24 = 25. g(x) degree is 50.
    # q(x) degree should be 25.
    try:
        q_poly, remainder = sp.div(g_poly, d_poly, domain=sp.QQ)
        # Check if division is exact? The problem implies it is.
        # If remainder is not 0, something is wrong, but theoretically it should be.
        # We proceed with q_poly.
    except Exception:
        return 1e100

    if _is_time_out():
        return 1e100

    # Find real roots of q(x)
    try:
        roots = sp.real_roots(q_poly)
    except Exception:
        return 1e100

    if not roots:
        return 0.0 # No real roots, B=0 (ideal)
        
    max_r = 0
    for r in roots:
        # r is a sympy RootOf or Rational
        # We need to check the sign condition:
        # q(r - 1e-198) and q(r + 1e-198) have strictly opposite signs.
        # This distinguishes simple roots from multiple roots or tangential roots.
        # Note: 1e-198 is very small.
        eps = sp.Rational(1, 10)**198 # 10^-198
        
        r_val = r  # symbolic
        
        # Evaluate q at r-eps and r+eps
        try:
            q_minus = q_poly.subs(x, r_val - eps)
            q_plus = q_poly.subs(x, r_val + eps)
            
            # Check signs
            # sympy numbers can be checked for sign
            if q_minus * q_plus < 0:
                # It is a valid crossing root
                r_float = float(r_val)
                if r_float > max_r:
                    max_r = r_float
        except Exception:
            continue

    return max_r

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    """
    # Initial guess: Zeros of J_{1/2}
    # J_{1/2}(x) ~ sin(sqrt(x))/sqrt(x)
    # Zeros at sqrt(x) = n*pi => x = (n*pi)^2
    z_init = [(i * math.pi) ** 2 for i in range(1, 13)]
    
    best_z = z_init[:]
    best_score = evaluate_B(best_z)
    
    # Optimization Loop
    # We use Nelder-Mead to minimize the score B.
    # Since the function is expensive, we limit the number of evaluations.
    # We also check time strictly.
    
    def objective(z_vec):
        # Ensure z_vec is 1D array of floats
        if len(z_vec) != 12:
            return 1e100
        # Small penalty for very small or non-positive values
        # The optimizer might drift.
        for z in z_vec:
            if z < 1e-3: # Keep away from 0
                return 1e100
        return evaluate_B(list(z_vec))
        
    # Run optimization
    # maxiter is a hard cap, but time check is the real governor.
    # Each eval takes ~5-30s. 
    # In 500s, we can afford maybe 20-50 evaluations.
    # Nelder-Mead with 12 dimensions needs at least 13 initial points.
    # So one full run might take 20-50 evals.
    
    try:
        res = minimize(
            objective, 
            z_init, 
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
            # Validate the result before accepting
            z_candidate = list(res.x)
            # Check if it's valid
            score_candidate = evaluate_B(z_candidate)
            if score_candidate < best_score and np.all(np.isfinite(z_candidate)) and np.all(np.array(z_candidate) > 0):
                # Check distinctness
                unique = np.unique(np.round(z_candidate, decimals=6))
                if len(unique) == 12:
                    best_z = z_candidate
                    best_score = score_candidate
    except Exception as e:
        pass # Ignore errors, return best so far
        
    if _is_time_out():
         return best_z
         
    return best_z
