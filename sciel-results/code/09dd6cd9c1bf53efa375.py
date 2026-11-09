import numpy as np
import sympy as sp
import time
from scipy.optimize import minimize
import mpmath as mp

def evaluate_candidate_exact(z_list):
    """
    Evaluates the candidate z_list exactly using sympy.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals
        z_rat = [sp.Rational(float(z)) for z in z_list]
        
        # Define x and Laguerre polynomials
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Precompute L_2k(x) for k=0..25
        Ls = []
        for k in range(26):
            Ls.append(sp.Laguerre(2*k, alpha, x))
        
        # Build matrix M and rhs directly
        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)
        
        # Eq 0: g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0
        
        # Eq 1: g'(0) = 1
        for k in range(26):
            M[1, k] = sp.diff(Ls[k], x).subs(x, 0)
        rhs[1] = 1
        
        idx = 2
        for z in z_rat:
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            
            for k in range(26):
                M[idx, k] = sp.diff(Ls[k], x).subs(x, z)
            rhs[idx] = 0
            idx += 1
        
        # Solve
        try:
            sol = M.LUsolve(rhs)
        except sp.matrices.exceptions.SingularMatrixException:
            return None
        except Exception:
            return None
        
        # Build g_sol
        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
        q = sp.div(g_sol, denom)
        q_poly = q[0]
        
        # Find real roots of q_poly
        try:
            roots = sp.real_roots(q_poly)
        except Exception:
            return None
            
        if not roots:
            return 0.0
            
        max_r = 0.0
        for r in roots:
            try:
                r_float = float(r)
            except:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            
            delta = 1e-198
            try:
                val_minus = float(q_poly.subs(x, r_float - delta))
                val_plus = float(q_poly.subs(x, r_float + delta))
            except:
                continue
            
            if val_minus * val_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def evaluate_candidate_fast(z_list):
    """
    Fast numerical evaluation of B using mpmath.
    Used for optimization. Returns B or a large penalty.
    """
    try:
        # Ensure z are positive and distinct
        z = sorted([float(z) for z in z_list])
        if any(z_i <= 0 for z_i in z):
            return 1e10
        for i in range(len(z) - 1):
            if z[i+1] - z[i] < 1e-6:
                return 1e10

        # Set precision for mpmath
        mp.mp.dps = 50
        
        alpha = mp.mpf(-1) / 2
        
        def gen_laguerre(n, alpha, x_val):
            total = mp.mpf(0)
            for j in range(n + 1):
                term = ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
                total += term
            return total
        
        def gen_laguerre_deriv(n, alpha, x_val):
            if n == 0:
                return mp.mpf(0)
            return -gen_laguerre(n-1, alpha+1, x_val)
        
        # Build M and rhs
        M = mp.matrix(26, 26)
        rhs = mp.matrix(26, 1)
        
        # g(0) = 0
        for k in range(26):
            M[0, k] = gen_laguerre(2*k, alpha, 0)
        rhs[0] = 0
        
        # g'(0) = 1
        for k in range(26):
            M[1, k] = gen_laguerre_deriv(2*k, alpha, 0)
        rhs[1] = 1
        
        idx = 2
        for z_val in z:
            for k in range(26):
                M[idx, k] = gen_laguerre(2*k, alpha, z_val)
            rhs[idx] = 0
            idx += 1
            
            for k in range(26):
                M[idx, k] = gen_laguerre_deriv(2*k, alpha, z_val)
            rhs[idx] = 0
            idx += 1
        
        # Solve
        try:
            sol = mp.lu_solve(M, rhs)
        except Exception:
            return 1e10
            
        # Construct g(x)
        def g(xv):
            total = mp.mpf(0)
            for k in range(26):
                total += sol[k] * gen_laguerre(2*k, alpha, xv)
            return total
            
        # Construct denom
        def denom(xv):
            d = xv
            for z_val in z:
                d *= (xv - z_val)**2
            return d
            
        # q(x) = g(x) / denom(x)
        def q(xv):
            return g(xv) / denom(xv)
            
        # Find roots of q
        # Scan for sign changes
        max_z = max(z)
        upper_bound = max_z * 2.0 + 100
        N_grid = 500
        xs = np.linspace(1e-6, upper_bound, N_grid)
        
        roots_found = []
        try:
            qs = [q(mp.mpf(xi)) for xi in xs]
        except Exception:
            return 1e10
            
        for i in range(N_grid - 1):
            if qs[i] * qs[i+1] < 0:
                try:
                    r = mp.findroot(lambda t: q(t), (mp.mpf(xs[i]), mp.mpf(xs[i+1])))
                    r_float = float(r)
                    if r_float > 0:
                        # Check if it's a z_i
                        is_z = False
                        for zi in z:
                            if abs(r_float - zi) < 1e-8:
                                is_z = True
                                break
                        if not is_z:
                            roots_found.append(r_float)
                except Exception:
                    pass
                    
        if not roots_found:
            return 1e10
            
        return max(roots_found)
        
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # Leave 60s margin
    
    # Strategy: 
    # The roots of Laguerre polynomials L_n^(alpha) scale roughly as n.
    # However, the optimal z_i for minimizing the largest root of q 
    # might follow a specific distribution.
    # We hypothesize that z_i ~ a + b * i^p.
    # Let's optimize the parameters (a, b, p) for the sequence z_i.
    # This reduces the 12-dimensional search to a 3-dimensional search.
    
    def generate_z(a, b, p):
        # i from 1 to 12
        i = np.arange(1, 13)
        z = a + b * (i ** p)
        # Ensure positive and sorted
        z = np.sort(z)
        if np.any(z <= 0):
            return None
        if len(set(np.round(z, 10))) < 12:
            return None
        return z

    def objective(params):
        a, b, p = params
        # Constraints: a > 0, b > 0, 0.5 < p < 2.0
        if a <= 0 or b <= 0 or p < 0.5 or p > 2.0:
            return 1e10
        
        z = generate_z(a, b, p)
        if z is None:
            return 1e10
            
        return evaluate_candidate_fast(z)
    
    best_B_exact = float('inf')
    best_z_exact = None
    
    # Multi-start local optimization on the 3 parameters (a, b, p)
    # Initial guesses based on asymptotic behavior
    # L_n zeros are approx (4n+2alpha+1)/4 + ...
    # Let's try a range of starts
    
    starts = [
        [1.0, 1.0, 1.0],
        [0.5, 2.0, 1.5],
        [2.0, 0.5, 1.0],
        [1.0, 1.5, 2.0],
        [0.1, 3.0, 1.0],
        [5.0, 0.1, 1.0]
    ]
    
    bounds = [(0.01, 10.0), (0.01, 10.0), (0.5, 2.0)]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(
                objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if res.fun < best_B_exact:
                a, b, p = res.x
                z_opt = generate_z(a, b, p)
                if z_opt is not None:
                    # Evaluate exactly to confirm
                    B_exact = evaluate_candidate_exact(z_opt)
                    if B_exact is not None and B_exact < best_B_exact:
                        best_B_exact = B_exact
                        best_z_exact = z_opt
        except Exception:
            pass

    # If the structured search didn't find a good candidate, 
    # fall back to a few random or heuristic candidates
    if best_z_exact is None:
        # Fallback: Bessel zeros
        bessel_zeros = [n * np.pi for n in range(1, 13)]
        B_exact = evaluate_candidate_exact(bessel_zeros)
        if B_exact is not None:
            best_B_exact = B_exact
            best_z_exact = bessel_zeros
        else:
            # Last resort
            return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    return best_z_exact
