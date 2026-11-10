import time
import math
import sympy as sp

_DEADLINE = 560.0  # seconds; keep margin under 600/630

def _to_rational_list(z_list):
    return [sp.Rational(float(z)) for z in z_list]

def _evaluate(z_list):
    """
    Evaluate B for a given list of 12 positive distinct doubles.
    Returns B (float) or None if invalid.
    """
    # Convert to rationals
    Z = _to_rational_list(z_list)
    
    # Check pairwise distinct and > 0
    for i in range(12):
        if Z[i] <= 0:
            return None
        for j in range(i+1, 12):
            if Z[i] == Z[j]:
                return None
    
    x = sp.Symbol('x')
    
    # Build generalized Laguerre polynomials L_k^{(-1/2)}(x) for k=0,2,...,50
    # g(x) = sum_{k=0}^{25} c_k L_{2k}^{(-1/2)}(x)
    # Actually the task says L_d^(alpha)(x) where alpha=-1/2, and g(x) = sum c_k L_{2k}(x)
    # The notation L_{2k}(x) likely means L_{2k}^{alpha}(x) with alpha=-1/2
    # Let's construct L_{2k}^{(-1/2)}(x)
    
    # We need equations:
    # g(0) = 0
    # g'(0) = 1
    # g(z_i) = 0 for all i
    # g'(z_i) = 0 for all i
    # Total: 2 + 12 + 12 = 26 equations for 26 unknowns c_0..c_25
    
    # Build the Laguerre polynomials
    def laguerre_generalized(k, alpha_val, x_val):
        """Compute L_k^{alpha}(x) symbolically"""
        return sp.laguerre(k, alpha_val, x_val)
    
    # c_k for k=0..25, corresponding to L_{2k}^{(-1/2)}(x)
    c = sp.symbols('c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25')
    
    # Precompute L_{2k}^{(-1/2)}(x)
    L_pols = []
    for k in range(26):
        L_pols.append(sp.laguerre(2*k, sp.Rational(-1, 2), x))
    
    # g(x) = sum c_k L_{2k}(x)
    g_expr = sum(c[k] * L_pols[k] for k in range(26))
    
    # g'(x)
    g_prime = sp.diff(g_expr, x)
    
    # Build equations
    eqs = []
    
    # g(0) = 0
    eqs.append(g_expr.subs(x, 0))
    
    # g'(0) = 1
    eqs.append(g_prime.subs(x, 0) - 1)
    
    # g(z_i) = 0 and g'(z_i) = 0
    for i in range(12):
        zi = Z[i]
        eqs.append(g_expr.subs(x, zi))
        eqs.append(g_prime.subs(x, zi) - 0)
    
    # Solve the linear system
    # Use sympy's linsolve or Matrix
    A = sp.Matrix([eq.subs(c, [1 if c == c[j] else 0 for j in range(26)]) for eq in eqs])
    # Actually, better to use sp.Matrix with the coefficients directly
    
    # Let's build the matrix properly
    A = sp.zeros(26, 26)
    b = sp.zeros(26, 1)
    
    # Equation 0: g(0) = 0
    g0 = g_expr.subs(x, 0)
    for k in range(26):
        A[0, k] = sp.simplify(g0.coeff(c[k]))
    b[0, 0] = 0
    
    # Equation 1: g'(0) = 1
    gprime0 = g_prime.subs(x, 0)
    for k in range(26):
        A[1, k] = sp.simplify(gprime0.coeff(c[k]))
    b[1, 0] = 1
    
    # Equations 2-25: g(z_i)=0, g'(z_i)=0
    eq_idx = 2
    for i in range(12):
        zi = Z[i]
        g_zi = g_expr.subs(x, zi)
        for k in range(26):
            A[eq_idx, k] = sp.simplify(g_zi.coeff(c[k]))
        b[eq_idx, 0] = 0
        eq_idx += 1
        
        gprime_zi = g_prime.subs(x, zi)
        for k in range(26):
            A[eq_idx, k] = sp.simplify(gprime_zi.coeff(c[k]))
        b[eq_idx, 0] = 0
        eq_idx += 1
    
    # Solve A * c = b
    try:
        c_sol = sp.LUsolve(A, b)
    except Exception:
        return None
    
    # Check if solution is valid (finite)
    for val in c_sol:
        if not sp.finite(val):
            return None
    
    # Substitute c into g
    c_dict = {c[k]: c_sol[k] for k in range(26)}
    g_final = g_expr.subs(c_dict)
    
    # q(x) = g(x) / (x * prod_i (x - z_i)^2)
    denom = x
    for i in range(12):
        denom *= (x - Z[i])**2
    
    # Exact polynomial division
    try:
        q = sp.div(g_final, denom, x)
        q_poly = q[0]  # quotient
        remainder = q[1]
        if remainder != 0:
            # Not a clean division, but let's proceed with quotient
            pass
    except Exception:
        return None
    
    if q_poly == 0:
        return 0.0
    
    # Find real roots of q
    try:
        real_roots = sp.real_roots(q_poly)
    except Exception:
        return 0.0
    
    # For each real root r, check sign change at r - 1e-198 and r + 1e-198
    delta = sp.Rational(1, 10)**sp.Rational(-198)
    # 1e-198 is extremely small; use it as a rational approximation
    # Actually 10^-198 is a rational number
    delta = sp.Rational(1, 10**198)
    
    max_r = 0
    for r in real_roots:
        r_num = sp.nsimplify(r)
        r_float = float(r_num)
        
        if r_float <= 0:
            continue
        
        # Evaluate q at r - delta and r + delta
        try:
            q_left = q_poly.subs(x, r - delta)
            q_right = q_poly.subs(x, r + delta)
            
            # Check for strictly opposite signs
            if q_left != 0 and q_right != 0:
                # Check sign
                left_positive = q_left > 0
                right_positive = q_right > 0
                if left_positive != right_positive:
                    if r_float > max_r:
                        max_r = r_float
        except Exception:
            continue
    
    return max_r

def construct():
    start_time = time.time()
    
    # Initial guesses
    guesses = [
        [4.0 * (i + 1) for i in range(12)],  # Seed: 4,8,...,48
        [1.0 * (i + 1) for i in range(12)],  # 1,2,...,12
        [2.0 * (i + 1) for i in range(12)],  # 2,4,...,24
        [0.5 * (i + 1) for i in range(12)],  # 0.5,1,...,6
        [0.8 * (i + 1) for i in range(12)],  # 0.8,1.6,...,9.6
        [1.5 * (i + 1) for i in range(12)],  # 1.5,3,...,18
        [3.0 * (i + 1) for i in range(12)],  # 3,6,...,36
    ]
    
    best_B = float('inf')
    best_z = None
    
    # Evaluate initial guesses
    for z in guesses:
        if time.time() - start_time > _DEADLINE:
            break
        B = _evaluate(z)
        if B is not None:
            if B < best_B:
                best_B = B
                best_z = list(z)
    
    # If no valid result yet, try a few more random-ish configurations
    if best_z is None:
        # Try evenly spaced with different scales
        for scale in [0.3, 0.6, 1.2, 2.5]:
            if time.time() - start_time > _DEADLINE:
                break
            z = [scale * (i + 1) for i in range(12)]
            B = _evaluate(z)
            if B is not None:
                if B < best_B:
                    best_B = B
                    best_z = list(z)
    
    # If we still have time, try local perturbation of the best solution
    if best_z is not None and time.time() - start_time < _DEADLINE - 120:
        # Try small perturbations
        import random
        random.seed(42)
        for attempt in range(20):
            if time.time() - start_time > _DEADLINE:
                break
            z_pert = list(best_z)
            # Perturb one random element
            idx = random.randint(0, 11)
            scale = z_pert[idx] * 0.2
            z_pert[idx] += random.uniform(-scale, scale)
            if z_pert[idx] <= 0:
                z_pert[idx] = abs(z_pert[idx]) + 0.1
            
            # Ensure distinctness
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    if abs(z_pert[i] - z_pert[j]) < 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            B = _evaluate(z_pert)
            if B is not None:
                if B < best_B:
                    best_B = B
                    best_z = list(z_pert)
    
    # Fallback: return seed if no better found
    if best_z is None:
        best_z = [4.0 * (i + 1) for i in range(12)]
    
    return best_z
