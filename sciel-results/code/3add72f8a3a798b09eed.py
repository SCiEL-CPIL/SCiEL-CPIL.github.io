"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
import sympy

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 625  # seconds, safety margin before 630s kill
    
    def elapsed():
        return time.time() - start_time
    
    def try_points(z_list):
        """Try a set of points and return (B_value, valid_flag)."""
        if elapsed() > DEADLINE - 45:
            return None, False
        
        # Validate: finite, > 0, pairwise distinct
        z_list = list(z_list)
        if len(z_list) != 12:
            return None, False
        for z in z_list:
            if not math.isfinite(z) or z <= 0:
                return None, False
        for i in range(12):
            for j in range(i+1, 12):
                if abs(z_list[i] - z_list[j]) < 1e-14:
                    return None, False
        
        # Convert to rationals and sort
        z_rationals = sorted(sympy.Rational(float(z)) for z in z_list)
        
        # Build the system
        x = sympy.Symbol('x')
        alpha = sympy.Rational(-1, 2)
        
        # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        coeffs = [sympy.Symbol(f'c{k}') for k in range(26)]
        
        # Precompute Laguerre polynomials
        L_pols = []
        for k in range(26):
            L = sympy.laguerre(2*k, x, alpha)
            L_pols.append(L)
        
        g = sum(coeffs[k] * L_pols[k] for k in range(26))
        g_prime = sympy.diff(g, x)
        
        # Build equations
        equations = []
        equations.append(sympy.Eq(g.subs(x, sympy.Integer(0)), sympy.Integer(0)))
        equations.append(sympy.Eq(g_prime.subs(x, sympy.Integer(0)), sympy.Integer(1)))
        
        for zi in z_rationals:
            equations.append(sympy.Eq(g.subs(x, zi), sympy.Integer(0)))
            equations.append(sympy.Eq(g_prime.subs(x, zi), sympy.Integer(0)))
        
        # Build linear system
        A = []
        b = []
        for eq in equations:
            expr = eq.lhs - eq.rhs
            row = []
            for c in coeffs:
                # Extract coefficient of c
                coeff_val = expr.coeff(c)
                row.append(coeff_val)
            A.append(row)
            # Constant term (set all c to 0)
            const = expr
            for c in coeffs:
                const = const.subs(c, sympy.Integer(0))
            b.append(-const)
        
        A_mat = sympy.Matrix(A)
        b_vec = sympy.Matrix(b)
        
        # Solve
        try:
            sol = A_mat.LUsolve(b_vec)
        except Exception:
            return None, False
        
        # Build g_final
        g_final = sum(sol[k] * L_pols[k] for k in range(26))
        
        # q(x) = g(x) / (x * prod_i (x - z_i)^2)
        denom = x
        for zi in z_rationals:
            denom *= (x - zi)**2
        
        try:
            q, rem = sympy.div(g_final, denom, domain='QQ')
        except Exception:
            return None, False
        
        if q == 0:
            return 0.0, True
        
        # Find real roots
        try:
            real_roots = sympy.real_roots(q)
        except Exception:
            return 0.0, True
        
        # Check sign changes
        counted_roots = []
        for r in real_roots:
            r_float = float(r)
            if not math.isfinite(r_float):
                continue
            # Use a small delta for sign check
            delta = sympy.Rational(1, 10)**198
            try:
                qm = float(q.subs(x, r - delta))
                qp = float(q.subs(x, r + delta))
                if math.isfinite(qm) and math.isfinite(qp) and qm * qp < 0:
                    counted_roots.append(r_float)
            except Exception:
                continue
        
        if counted_roots:
            B_val = max(0.0, max(counted_roots))
        else:
            B_val = 0.0
        
        return B_val, True
    
    # Try candidate configurations
    # Strategy: cluster points near 0 to minimize the largest root of q
    
    # Candidate 1: Small arithmetic progression
    cand1 = [1e-4 * (i + 1) for i in range(12)]
    
    # Candidate 2: Even smaller
    cand2 = [1e-5 * (i + 1) for i in range(12)]
    
    # Candidate 3: Geometric
    cand3 = [1e-6 * (2.0**i) for i in range(12)]
    
    # Candidate 4: Moderate spread
    cand4 = [0.05 * (i + 1) for i in range(12)]
    
    # Candidate 5: Tight cluster
    cand5 = [1e-7 * (i + 1) for i in range(12)]
    
    candidates = [cand1, cand2, cand3, cand4, cand5]
    
    best_B = float('inf')
    best_z = cand1[:]
    
    for idx, cand in enumerate(candidates):
        if elapsed() > DEADLINE - 50:
            break
        
        B_val, valid = try_points(cand)
        
        if valid and B_val is not None:
            if B_val < best_B:
                best_B = B_val
                best_z = cand[:]
        
        print(f"Candidate {idx}: B={B_val}, valid={valid}, elapsed={elapsed():.1f}s")
    
    return best_z
# EVOLVE-BLOCK-END
