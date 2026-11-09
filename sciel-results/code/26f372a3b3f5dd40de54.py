import sympy as sp
import numpy as np
import time
import math

def _gen_laguerre(n, alpha, x_val):
    """Compute L_n^{(alpha)}(x) using explicit sum (for mpmath)."""
    import mpmath as mp
    total = mp.mpf(0)
    for j in range(n + 1):
        total += ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
    return total

def _gen_laguerre_deriv(n, alpha, x_val):
    """d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)."""
    import mpmath as mp
    if n == 0:
        return mp.mpf(0)
    return -_gen_laguerre(n - 1, alpha + 1, x_val)

def evaluate_candidate_exact(z_list):
    """
    Exact sympy evaluation. Returns B (float) or None if invalid.
    """
    try:
        z_rat = [sp.Rational(float(z)) for z in z_list]
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)

        Ls = [sp.Laguerre(2 * k, alpha, x) for k in range(26)]

        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)

        # g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0

        # g'(0) = 1
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

        try:
            sol = M.LUsolve(rhs)
        except Exception:
            return None

        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))

        denom = x
        for z in z_rat:
            denom *= (x - z) ** 2

        q, rem = sp.div(g_sol, denom)
        if rem != 0:
            return None

        try:
            roots = sp.real_roots(q)
        except Exception:
            return None

        if not roots:
            return 0.0

        max_r = 0.0
        for r in roots:
            try:
                r_float = float(r)
            except Exception:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            delta = 1e-198
            try:
                v_minus = float(q.subs(x, r_float - delta))
                v_plus = float(q.subs(x, r_float + delta))
            except Exception:
                continue
            if v_minus * v_plus < 0:
                if r_float > max_r:
                    max_r = r_float
        return float(max_r)
    except Exception:
        return None

def evaluate_candidate_fast(z_list):
    """
    Fast mpmath evaluation. Returns B estimate or large penalty.
    """
    try:
        import mpmath as mp
        z = sorted([float(zz) for zz in z_list])
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(len(z) - 1):
            if z[i + 1] - z[i] < 1e-6:
                return 1e10

        mp.mp.dps = 40
        alpha = mp.mpf(-1) / 2

        # Precompute Ls values and derivatives at z's for matrix construction
        M = mp.matrix(26, 26)
        rhs = mp.matrix(26, 1)

        for k in range(26):
            M[0, k] = _gen_laguerre(2 * k, alpha, mp.mpf(0))
        rhs[0] = 0

        for k in range(26):
            M[1, k] = _gen_laguerre_deriv(2 * k, alpha, mp.mpf(0))
        rhs[1] = 1

        idx = 2
        for zz in z:
            zv = mp.mpf(zz)
            for k in range(26):
                M[idx, k] = _gen_laguerre(2 * k, alpha, zv)
            rhs[idx] = 0
            idx += 1
            for k in range(26):
                M[idx, k] = _gen_laguerre_deriv(2 * k, alpha, zv)
                rhs[idx] = 0
            idx += 1

        try:
            sol = mp.lu_solve(M, rhs)
        except Exception:
            return 1e10

        def g_val(xv):
            total = mp.mpf(0)
            for k in range(26):
                total += sol[k] * _gen_laguerre(2 * k, alpha, xv)
            return total

        def denom_val(xv):
            d = xv
            for zz in z:
                d *= (xv - mp.mpf(zz)) ** 2
            return d

        def q_val(xv):
            return g_val(xv) / denom_val(xv)

        max_z = max(z)
        upper_bound = max_z * 1.8 + 80

        # Coarse scan for sign changes
        N_grid = 2000
        xs = np.linspace(1e-5, upper_bound, N_grid)
        prev_q = None
        roots_found = []

        for xi in xs:
            xv = mp.mpf(xi)
            try:
                qi = q_val(xv)
            except Exception:
                prev_q = None
                continue
            if prev_q is not None and prev_q * qi < 0:
                # Bisect / findroot
                a = mp.mpf(xs[list(xs).index(xi) - 1]) if list(xs).index(xi) > 0 else mp.mpf(1e-5)
                b = mp.mpf(xi)
                try:
                    r = mp.findroot(lambda t: q_val(t), (a, b))
                    r_f = float(r)
                    if 0 < r_f < upper_bound:
                        is_z = any(abs(r_f - zi) < 1e-9 for zi in z)
                        if not is_z:
                            roots_found.append(r_f)
                except Exception:
                    pass
            prev_q = qi

        if not roots_found:
            # Try a broader scan
            for xi in np.linspace(1e-5, upper_bound * 2, 500):
                xv = mp.mpf(xi)
                try:
                    qi = q_val(xv)
                except Exception:
                    continue
                if prev_q is not None and prev_q * qi < 0:
                    try:
                        r = mp.findroot(lambda t: q_val(t), (mp.mpf(xi * 0.99), xv))
                        r_f = float(r)
                        if 0 < r_f:
                            is_z = any(abs(r_f - zi) < 1e-9 for zi in z)
                            if not is_z:
                                roots_found.append(r_f)
                    except Exception:
                        pass
                prev_q = qi

        if not roots_found:
            return 1e10
        return max(roots_found)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin

    # We use a bisection strategy on the target B value.
    # We want to find z_i such that the largest real root of q(x) is <= B_target.
    # Actually, the problem is: given z_i, B is determined. We want to minimize B.
    # This is equivalent to finding the smallest B such that there exist z_i 
    # satisfying the conditions and B(z_i) <= B.
    
    # However, solving for z_i given B is a non-linear system.
    # Let's define the system:
    # We have 12 z_i. 
    # The condition is that g(x) has double roots at z_i.
    # g(x) = sum c_k L_{2k}(x).
    # The coefficients c_k are determined by z_i (26 equations, 26 unknowns c_k).
    # So for any z_i, there is a unique g (if matrix is invertible).
    # Then q(x) is determined. B is the max real root of q.
    
    # Strategy: 
    # 1. Use the Bessel zero squares as a good initial guess for z_i.
    # 2. We know that shifting z_i slightly inward reduces B.
    # 3. Instead of Nelder-Mead, we use Sympy's nsolve to solve for z_i 
    #    such that q(B_target) = 0 and q'(B_target) = 0? No, that's not right.
    #    The root B is not a double root of q. It's a simple root (counted by sign change).
    #    
    # Let's rethink. The current best score is ~44. B = score * 2*pi ~ 276.
    # The Bessel zeros squared are around 5.7 to 1363.
    # The largest z is ~1363. B is ~276. This means B is much smaller than max(z).
    # This implies the "spillover" root is determined by the global shape, not the largest node.
    
    # A better structural approach:
    # We can optimize the nodes z_i directly using Sympy's numerical solver 
    # on the objective function B(z). But B(z) is non-smooth (max of roots).
    # However, for a fixed configuration, the largest root is usually isolated.
    # We can use gradient-free methods but with a better starting point and 
    # more robust handling.
    
    # The previous program used Nelder-Mead on a fast mpmath evaluator.
    # The issue might be that the fast evaluator is not accurate enough or 
    # the optimization got stuck.
    
    # New Strategy: 
    # 1. Start with Bessel zeros squared.
    # 2. Use Sympy's `minimize` with a numerical gradient? No, Sympy minimize is slow.
    # 3. Use `scipy.optimize.differential_evolution` on the fast evaluator? 
    #    It's global, but might be too slow for 12 dimensions.
    # 4. Use `scipy.optimize.dual_annealing`?
    
    # Let's try a different angle: 
    # The nodes z_i are close to the zeros of some orthogonal polynomial.
    # For alpha = -1/2, the Laguerre polynomials are related to Bessel functions.
    # The zeros of L_n^{(-1/2)} are related to Bessel zeros.
    # Specifically, the zeros of L_n^{(-1/2)}(x) are approximately (j_{-1/2, k})^2 / 4?
    # Actually, for large n, the zeros of L_n^{(alpha)}(x) are approximately 
    # 4n x_k where x_k are zeros of J_alpha.
    # Here we have a sum of L_{2k}.
    
    # Let's stick to the optimization but improve the starting point and 
    # use a more robust optimizer like 'L-BFGS-B' if we can get gradients, 
    # or 'Powell' which is derivative-free and robust.
    
    # The previous program used Nelder-Mead. Let's try Powell.
    # Also, we will run multiple restarts from slightly different starting points.
    
    # Starting points:
    # 1. Bessel J0 zeros squared.
    # 2. Bessel J1 zeros squared?
    # 3. Uniform distribution in [1, 1000]?
    
    # We will use the fast evaluator for the optimization.
    # We will use the exact evaluator only for the final verification.
    
    j0_zeros = [
        2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
        14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
        27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
    ]
    bessel_sq = [z * z for z in j0_zeros]
    
    # Try scaling the Bessel zeros
    # The previous best was ~44. 
    # Let's try scaling factors 0.8 to 1.2.
    
    start_points = []
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        start_points.append([z * scale for z in bessel_sq])
        
    # Also try adding a linear shift? No, they must be positive.
    
    best_B = float('inf')
    best_z = None
    
    from scipy.optimize import minimize
    
    for start in start_points:
        if time.time() > deadline - 100:
            break
            
        # Use Powell method, which is robust for non-smooth functions
        try:
            res = minimize(evaluate_candidate_fast, start, method='Powell',
                           options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-6})
            
            if res.fun < best_B:
                # Verify with exact evaluator
                z_opt = sorted(res.x)
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 8) for zi in z_opt)) == 12:
                    B_exact = evaluate_candidate_exact(z_opt)
                    if B_exact is not None and B_exact < best_B:
                        best_B = B_exact
                        best_z = z_opt
        except Exception:
            pass

    if best_z is None:
        # Fallback to original Bessel zeros
        return bessel_sq
        
    return best_z
