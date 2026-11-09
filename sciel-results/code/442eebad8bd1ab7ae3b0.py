import numpy as np
import sympy as sp
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import genlaguerre
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
        x = mp.var('x')
        
        # Precompute Laguerre polynomials L_2k
        Ls = []
        for k in range(26):
            # mpmath's laguerre is L_n(x)
            # We need L_{2k}^{alpha}(x). mpmath.laguerre(n, x) is standard.
            # For generalized, we might need to implement or use sympy's logic in mpmath? 
            # No, sympy is slow. 
            # mpmath doesn't have generalized Laguerre directly in a simple call?
            # Actually, mpmath.laguerre(n, x) is standard. 
            # We can use the recurrence or explicit formula.
            # L_n^{(alpha)}(x) = sum_{j=0}^n (-1)^j * binomial(n+alpha, n-j) * x^j / j!
            
            def gen_laguerre(n, alpha, x_val):
                total = mp.mpf(0)
                for j in range(n + 1):
                    term = ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
                    total += term
                return total
            
            # We need to build the system. 
            # This is too slow if we do it inside the optimization loop repeatedly.
            # We should precompute the basis functions as mpmath functions.
            
            Ls.append(lambda xv, n=2*k, a=alpha: gen_laguerre(n, a, xv))

        # Build M and rhs
        M = mp.matrix(26, 26)
        rhs = mp.matrix(26, 1)
        
        # g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k](0)
        rhs[0] = 0
        
        # g'(0) = 1
        # Derivative of L_n^{(alpha)}(x) is -L_{n-1}^{(alpha+1)}(x)
        # So we need L_{2k-1}^{(1/2)}(x) for k>=1, and for k=0 it's 0.
        # Wait, derivative of L_0 is 0.
        # Let's just use numerical differentiation or analytic formula.
        # Analytic: d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)
        
        def gen_laguerre_deriv(n, alpha, x_val):
            if n == 0:
                return mp.mpf(0)
            return -gen_laguerre(n-1, alpha+1, x_val)
        
        for k in range(26):
            M[1, k] = gen_laguerre_deriv(2*k, alpha, 0)
        rhs[1] = 1
        
        idx = 2
        for z in z:
            for k in range(26):
                M[idx, k] = Ls[k](z)
            rhs[idx] = 0
            idx += 1
            
            for k in range(26):
                M[idx, k] = gen_laguerre_deriv(2*k, alpha, z)
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
                total += sol[k] * Ls[k](xv)
            return total
            
        # Construct denom
        def denom(xv):
            d = xv
            for z in z:
                d *= (xv - z)**2
            return d
            
        # q(x) = g(x) / denom(x)
        # We need to find roots of q(x). 
        # Since g has double roots at z_i and denom has double roots at z_i, 
        # q is a polynomial. 
        # We can find roots of q by finding roots of g that are NOT z_i? 
        # No, q is a polynomial of degree 50 - (1 + 24) = 25.
        # Wait, g is degree 50. denom is degree 1 + 24 = 25.
        # So q is degree 25.
        
        # We can evaluate q(x) directly for large x to find the largest root?
        # Or we can construct the polynomial q explicitly?
        # Polynomial division in mpmath is not built-in for high precision easily.
        
        # Alternative: Use the fact that q(x) = g(x) / denom(x).
        # The roots of q are the roots of g that are not z_i.
        # But g has double roots at z_i. 
        # So if we find all real roots of g, we remove the z_i's (which are double).
        # The remaining roots are the roots of q.
        
        # How to find roots of g?
        # g is a sum of Laguerre polynomials.
        # We can use mpmath.findroot? It's local.
        # We need global roots.
        
        # Let's try to evaluate q(x) on a grid and find sign changes.
        # Then refine with findroot.
        
        def q(xv):
            return g(xv) / denom(xv)
            
        # The largest root B is likely the largest root of g that is not a z_i.
        # Since g(x) behaves like x^50 and denom like x^25, q(x) ~ x^25.
        # So q(x) -> infinity as x -> infinity.
        # We need to find the largest real root of q.
        
        # Strategy: Scan for sign changes of q(x) in a wide range.
        # Range: [0, max(z)*2 + 100] ?
        # The roots of Laguerre polynomials are positive.
        
        max_z = max(z)
        upper_bound = max_z * 1.5 + 50
        
        # Grid search
        roots_found = []
        N_grid = 1000
        xs = np.linspace(1e-6, upper_bound, N_grid)
        qs = [q(mp.mpf(xi)) for xi in xs]
        
        # Find sign changes
        for i in range(N_grid - 1):
            if qs[i] * qs[i+1] < 0:
                # Root in [xs[i], xs[i+1]]
                try:
                    r = mp.findroot(lambda t: q(t), (mp.mpf(xs[i]), mp.mpf(xs[i+1])))
                    if r > 0 and r < upper_bound:
                        # Check if it's a z_i
                        is_z = False
                        for zi in z:
                            if abs(r - zi) < 1e-10:
                                is_z = True
                                break
                        if not is_z:
                            roots_found.append(float(r))
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
    
    # Initial candidates
    # Use Bessel J_{1/2} zeros: pi, 2pi, ..., 12pi
    bessel_zeros = [n * np.pi for n in range(1, 13)]
    
    # Also try a slightly scaled version
    bessel_scaled = [n * np.pi * 0.9 for n in range(1, 13)]
    
    candidates = [bessel_zeros, bessel_scaled]
    
    best_B_exact = float('inf')
    best_z_exact = None
    
    # Evaluate initial candidates exactly to have a baseline
    for z in candidates:
        if time.time() > deadline - 30:
            break
        B = evaluate_candidate_exact(z)
        if B is not None and B < best_B_exact:
            best_B_exact = B
            best_z_exact = z
    
    # Optimization
    # We will run Differential Evolution on the fast evaluator.
    # DE is global and robust.
    
    def objective(z):
        return evaluate_candidate_fast(z)
    
    # Bounds for DE
    # z_i should be positive and roughly ordered.
    # We can just let DE search in a box and sort/penalize in objective.
    # Bounds: [1, 100] for each z_i
    bounds = [(1, 100)] * 12
    
    # Run DE with limited time
    de_start_time = time.time()
    de_time_limit = 300  # 5 minutes
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=True  # Use L-BFGS-B for final polish
        )
        
        if res.fun < best_B_exact:
            z_opt = sorted(res.x)
            if all(zi > 0 for zi in z_opt) and len(set([round(zi, 8) for zi in z_opt])) == 12:
                B_exact = evaluate_candidate_exact(z_opt)
                if B_exact is not None and B_exact < best_B_exact:
                    best_B_exact = B_exact
                    best_z_exact = z_opt
    except Exception:
        pass
    
    # If time remains, do a local refinement from the best point found
    if time.time() < deadline - 60 and best_z_exact is not None:
        try:
            res_local = minimize(
                objective, 
                best_z_exact, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res_local.fun < best_B_exact:
                z_opt = sorted(res_local.x)
                if all(zi > 0 for zi in z_opt) and len(set([round(zi, 8) for zi in z_opt])) == 12:
                    B_exact = evaluate_candidate_exact(z_opt)
                    if B_exact is not None and B_exact < best_B_exact:
                        best_B_exact = B_exact
                        best_z_exact = z_opt
        except Exception:
            pass

    if best_z_exact is None:
        return bessel_zeros
        
    return best_z_exact
