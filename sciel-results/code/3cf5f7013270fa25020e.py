"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    def score_z(z_list):
        """Compute B for a given list of 12 double roots. Returns B or inf if invalid."""
        if len(z_list) != 12:
            return float('inf')
        # Check finite, > 0, pairwise distinct
        for z in z_list:
            if not np.isfinite(z) or z <= 0:
                return float('inf')
        for i in range(12):
            for j in range(i+1, 12):
                if abs(z_list[i] - z_list[j]) < 1e-15:
                    return float('inf')
        
        try:
            import sympy as sp
            from sympy import Rational, symbols, Poly, laguerre, diff, solve, nsolve, real_roots
            
            x = symbols('x')
            alpha = Rational(-1, 2)
            m = 12
            n = 1
            
            # Build the system of equations
            # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
            # L_d^{alpha}(x) is the generalized Laguerre polynomial of degree d with parameter alpha
            
            # We need to solve for c_0, ..., c_25 such that:
            # g(0) = 0
            # g'(0) = 1
            # g(z_i) = 0 for all i
            # g'(z_i) = 0 for all i
            # Total: 2 + 12 + 12 = 26 equations, 26 unknowns
            
            # Convert z values to rationals
            z_rationals = [Rational(float(z)) for z in z_list]
            z_rationals = sorted(z_rationals)
            
            # Build the coefficient matrix and RHS
            # Row 0: g(0) = 0
            # Row 1: g'(0) = 1
            # Rows 2-13: g(z_i) = 0
            # Rows 14-25: g'(z_i) = 0
            
            # Precompute L_{2k}^{alpha}(x) for k=0..25
            # L_d^{alpha}(x) = sum_{j=0}^{d} binom(d+alpha, d-j) * (-1)^j * x^j / j!
            # But sympy's laguerre(d, alpha, x) gives this directly
            
            # For efficiency, let's build the system using sympy's linear algebra
            # This will be slow but exact
            
            # Actually, let's use a more efficient approach:
            # Build the 26x26 matrix and solve with LUsolve
            
            # First, compute L_{2k}^{alpha}(0) and (L_{2k}^{alpha})'(0) for k=0..25
            # L_d^{alpha}(0) = binom(d+alpha, d)
            # (L_d^{alpha})'(0) = -binom(d+alpha, d-1) for d >= 1, 0 for d=0
            
            from sympy import binom
            
            c_syms = [symbols(f'c{k}') for k in range(26)]
            
            # Build equations
            eqs = []
            mat = sp.zeros(26, 26)
            rhs = sp.zeros(26, 1)
            
            # g(0) = 0
            for k in range(26):
                d = 2 * k
                val = laguerre(d, alpha, 0)
                mat[0, k] = val
            rhs[0, 0] = 0
            
            # g'(0) = 1
            for k in range(26):
                d = 2 * k
                if d == 0:
                    val = 0
                else:
                    val = diff(laguerre(d, alpha, x), x).subs(x, 0)
                mat[1, k] = val
            rhs[1, 0] = 1
            
            # g(z_i) = 0 for each i
            for i, zi in enumerate(z_rationals):
                row = 2 + i
                for k in range(26):
                    d = 2 * k
                    val = laguerre(d, alpha, zi)
                    mat[row, k] = val
                rhs[row, 0] = 0
                
                row2 = 14 + i
                for k in range(26):
                    d = 2 * k
                    val = diff(laguerre(d, alpha, x), x).subs(x, zi)
                    mat[row2, k] = val
                rhs[row2, 0] = 0
            
            # Solve the linear system
            try:
                sol = sp.LUsolve(mat, rhs)
            except Exception:
                return float('inf')
            
            # Build g(x)
            g = sum(sol[k, 0] * laguerre(2*k, alpha, x) for k in range(26))
            g = sp.expand(g)
            
            # Compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
            denom = x
            for zi in z_rationals:
                denom = denom * (x - zi)**2
            denom = sp.expand(denom)
            
            # Polynomial division
            try:
                q, rem = sp.div(g, denom, x)
            except Exception:
                return float('inf')
            
            if rem != 0:
                return float('inf')
            
            # Find real roots of q
            try:
                roots = real_roots(q, x)
            except Exception:
                return float('inf')
            
            B = 0
            for r in roots:
                # Check if r is strictly positive and if q has opposite signs at r-eps and r+eps
                if r <= 0:
                    continue
                eps = 10**(-198)
                r_minus = r - eps
                r_plus = r + eps
                q_minus = q.subs(x, r_minus)
                q_plus = q.subs(x, r_plus)
                
                # Check for opposite signs
                if q_minus == 0 or q_plus == 0:
                    continue
                if (q_minus > 0 and q_plus < 0) or (q_minus < 0 and q_plus > 0):
                    if r > B:
                        B = r
            
            return float(B)
        
        except Exception:
            return float('inf')
    
    # Coarse grid search over z_1 and r
    best_B = float('inf')
    best_z = None
    
    # Log-spaced grid for z_1 and r
    z1_values = np.logspace(-2, 2, 15)  # 0.01 to 100
    r_values = np.logspace(-0.5, 0.5, 15)  # ~0.316 to ~3.16
    
    for z1 in z1_values:
        if time.time() > deadline - 60:
            break
        for r in r_values:
            if time.time() > deadline - 30:
                break
            
            z_list = [z1 * (r ** i) for i in range(12)]
            
            # Check if all positive and finite
            if any(not np.isfinite(z) or z <= 0 for z in z_list):
                continue
            
            B = score_z(z_list)
            
            if B < best_B:
                best_B = B
                best_z = z_list.copy()
                print(f"New best: B={B}, z1={z1}, r={r}")
    
    if best_z is not None:
        return best_z
    else:
        # Fallback: return some valid distinct positive numbers
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
# EVOLVE-BLOCK-END
