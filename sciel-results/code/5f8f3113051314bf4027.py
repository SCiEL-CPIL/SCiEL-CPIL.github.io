"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """Return 12 double-root locations z_1..z_12 using iterative optimization."""
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 45.0

    def time_remaining():
        return deadline - time.time() - safety_margin

    def compute_score(z_list):
        """Compute the exact score B/(2*pi) for given z locations."""
        import sympy as sp
        
        # Convert to rationals
        z_rat = [sp.Rational(float(z)) for z in z_list]
        z_rat_sorted = sorted(z_rat)
        
        # Build g(x) = sum c_k L_2k(x) for k=0..25
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Laguerre polynomials L_2k^(alpha)(x)
        lag_terms = []
        for k in range(26):
            deg = 2 * k
            L = sp.genlaguerre(deg, alpha)
            lag_terms.append(L(x))
        
        # Coefficients c_0..c_25
        c = sp.symbols('c0:26')
        g = sum(c[k] * lag_terms[k] for k in range(26))
        
        # Build linear system: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for all i
        eqs = []
        # g(0) = 0
        eqs.append(g.subs(x, 0))
        # g'(0) = 1
        g_diff = sp.diff(g, x)
        eqs.append(g_diff.subs(x, 0) - 1)
        # For each z_i: g(z_i)=0, g'(z_i)=0
        for z in z_rat_sorted:
            eqs.append(g.subs(x, z))
            eqs.append(g_diff.subs(x, z))
        
        # Solve the linear system
        A = sp.Matrix([[eq.subs(c[j], 0) if i != j else 1 for j in range(26)] for i in range(26)])
        # Actually, need to build the system properly
        # The system is: for each equation, the coefficients of c_0..c_25
        A = sp.Matrix(26, 26, lambda i, j: 0)
        b = sp.Matrix(26, 1, lambda i, j: 0)
        
        for i, eq in enumerate(eqs):
            # eq is a linear expression in c_0..c_25
            for j in range(26):
                A[i, j] = eq.coeff(c[j])
            # Move constant terms to b
            b[i] = -eq.subs({c[j]: 0 for j in range(26)})
        
        try:
            c_sol = A.LUsolve(b)
        except Exception:
            return float('inf')
        
        # Substitute solution into g
        g_sol = g.subs({c[j]: c_sol[j] for j in range(26)})
        
        # q(x) = g(x) / (x * prod_i (x - z_i)^2)
        denom = x
        for z in z_rat_sorted:
            denom *= (x - z)**2
        
        try:
            q = sp.simplify(sp.together(g_sol / denom))
            # Check if it's a polynomial
            if not q.is_polynomial(x):
                return float('inf')
            q_poly = sp.Poly(q, x)
        except Exception:
            return float('inf')
        
        # Find real roots of q
        try:
            real_roots = sp.real_roots(q_poly)
        except Exception:
            return float('inf')
        
        # Count roots with sign change at r +/- 1e-198
        eps = sp.Rational(1, 10**198)
        B = 0
        for r in real_roots:
            try:
                val_left = q_poly.eval(r - eps)
                val_right = q_poly.eval(r + eps)
                if val_left * val_right < 0:
                    if r > B:
                        B = r
            except Exception:
                continue
        
        return float(B) / (2 * math.pi)

    def validate_z(z_list):
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

    # Initial guesses: Bessel-based, evenly spaced, and a few random
    initial_guesses = []
    
    # Bessel J_{1/2} zeros squared
    initial_guesses.append([float(k**2 * math.pi**2) for k in range(1, 13)])
    
    # Evenly spaced
    initial_guesses.append([float(i * 10.0) for i in range(1, 13)])
    
    # Logarithmically spaced
    initial_guesses.append([float(10 ** (0.1 + 0.1 * i)) for i in range(12)])
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        z_rand = sorted(np.random.uniform(0.5, 200, 12).tolist())
        initial_guesses.append(z_rand)
    
    best_z = None
    best_score = float('inf')
    
    # Try each initial guess
    for guess in initial_guesses:
        if time_remaining() < 60:
            break
        
        if not validate_z(guess):
            continue
        
        try:
            score = compute_score(guess)
            if score < best_score:
                best_score = score
                best_z = guess[:]
        except Exception:
            continue
    
    # If we found a good point, do local refinement with Nelder-Mead
    if best_z is not None and time_remaining() > 120:
        from scipy.optimize import minimize
        
        def objective(z_flat):
            z_list = list(z_flat)
            if not validate_z(z_list):
                return 1e10
            try:
                return compute_score(z_list)
            except Exception:
                return 1e10
        
        # Nelder-Mead
        try:
            res = minimize(objective, best_z, method='Nelder-Mead',
                          options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success or res.fun < best_score:
                z_res = list(res.x)
                if validate_z(z_res):
                    score_res = compute_score(z_res)
                    if score_res < best_score:
                        best_score = score_res
                        best_z = z_res
        except Exception:
            pass
    
    # Final validation
    if best_z is not None and validate_z(best_z):
        return best_z
    
    # Fallback
    return [float(i * 10.0) for i in range(1, 13)]

# EVOLVE-BLOCK-END
