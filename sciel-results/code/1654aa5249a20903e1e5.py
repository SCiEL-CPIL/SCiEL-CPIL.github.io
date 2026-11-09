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

        N_grid = 2000
        xs = np.linspace(1e-5, upper_bound, N_grid)
        prev_q = None
        roots_found = []

        for i in range(1, N_grid):
            xi = xs[i]
            xv = mp.mpf(xi)
            try:
                qi = q_val(xv)
            except Exception:
                prev_q = None
                continue
            if prev_q is not None and prev_q * qi < 0:
                a = mp.mpf(xs[i - 1])
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
    deadline = start_time + 550.0  # Safety margin for exact evaluation

    # Structural change: Instead of Bessel zeros (which give score ~44),
    # we use a different parameterization and optimization strategy.
    # 
    # Key insight: The problem is equivalent to finding 12 nodes z_i such that
    # the "spillover" root B is minimized. The Bessel zeros are natural for
    # orthogonal polynomials but may not be optimal for this specific constraint.
    #
    # New approach: Use a geometric progression parameterization with two parameters
    # (ratio and scale), optimized in 2D instead of 12D. This dramatically reduces
    # the search space and allows more thorough exploration.
    #
    # Additionally, we try multiple families:
    # 1. Geometric: z_i = A * r^i
    # 2. Arithmetic: z_i = A + i*d
    # 3. Bessel with perturbation
    # 4. Power law: z_i = A * i^p
    
    from scipy.optimize import minimize, differential_evolution
    
    best_B = float('inf')
    best_z = None
    
    # Family 1: Geometric progression z_i = A * r^i for i=0..11
    # Parameters: log(A), log(r) to ensure positivity
    def objective_geo(params):
        log_A, log_r = params
        A = np.exp(log_A)
        r = np.exp(log_r)
        z = [A * (r ** i) for i in range(12)]
        # Ensure distinct and positive
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(11):
            if abs(z[i+1] - z[i]) < 1e-8:
                return 1e10
        return evaluate_candidate_fast(z)
    
    # Try several initial points for geometric
    geo_inits = [
        [math.log(5.0), math.log(1.4)],
        [math.log(10.0), math.log(1.3)],
        [math.log(2.0), math.log(1.8)],
        [math.log(50.0), math.log(1.2)],
        [math.log(1.0), math.log(2.0)],
        [math.log(0.5), math.log(2.5)],
        [math.log(100.0), math.log(1.1)],
    ]
    
    for init in geo_inits:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(objective_geo, init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_B:
                log_A, log_r = res.x
                A = np.exp(log_A)
                r = np.exp(log_r)
                z_opt = sorted([A * (r ** i) for i in range(12)])
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 10) for zi in z_opt)) == 12:
                    B_val = res.fun
                    if B_val < best_B:
                        best_B = B_val
                        best_z = z_opt
        except Exception:
            pass
    
    # Family 2: Arithmetic progression z_i = A + i*d
    def objective_arith(params):
        A, d = params
        if A <= 0 or d <= 0:
            return 1e10
        z = [A + i * d for i in range(12)]
        return evaluate_candidate_fast(z)
    
    arith_inits = [
        [5.0, 20.0],
        [10.0, 15.0],
        [2.0, 30.0],
        [20.0, 10.0],
        [1.0, 40.0],
    ]
    
    for init in arith_inits:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(objective_arith, init, method='Nelder-Mead',
                           bounds=[(0.1, 500.0), (0.1, 200.0)],
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_B:
                A, d = res.x
                z_opt = sorted([A + i * d for i in range(12)])
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 10) for zi in z_opt)) == 12:
                    B_val = res.fun
                    if B_val < best_B:
                        best_B = B_val
                        best_z = z_opt
        except Exception:
            pass
    
    # Family 3: Power law z_i = A * (i+1)^p
    def objective_power(params):
        log_A, p = params
        A = np.exp(log_A)
        z = [A * ((i + 1) ** p) for i in range(12)]
        if any(zi <= 0 for zi in z):
            return 1e10
        for i in range(11):
            if abs(z[i+1] - z[i]) < 1e-8:
                return 1e10
        return evaluate_candidate_fast(z)
    
    power_inits = [
        [math.log(5.0), 1.5],
        [math.log(10.0), 1.0],
        [math.log(2.0), 2.0],
        [math.log(50.0), 0.5],
        [math.log(1.0), 3.0],
    ]
    
    for init in power_inits:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(objective_power, init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_B:
                log_A, p = res.x
                A = np.exp(log_A)
                z_opt = sorted([A * ((i + 1) ** p) for i in range(12)])
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 10) for zi in z_opt)) == 12:
                    B_val = res.fun
                    if B_val < best_B:
                        best_B = B_val
                        best_z = z_opt
        except Exception:
            pass
    
    # Family 4: Bessel zeros with global perturbation
    j0_zeros = [
        2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
        14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
        27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
    ]
    bessel_base = [z * z for z in j0_zeros]
    
    def objective_bessel(params):
        # Scale and shift the Bessel nodes
        scale, shift = params
        if scale <= 0:
            return 1e10
        z = [max(0.1, (bz * scale) + shift) for bz in bessel_base]
        # Ensure distinct
        for i in range(11):
            if abs(z[i+1] - z[i]) < 1e-8:
                return 1e10
        return evaluate_candidate_fast(z)
    
    bessel_inits = [
        [1.0, 0.0],
        [0.9, 5.0],
        [1.1, -5.0],
        [0.8, 10.0],
        [1.2, -10.0],
    ]
    
    for init in bessel_inits:
        if time.time() > deadline - 120:
            break
        try:
            res = minimize(objective_bessel, init, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_B:
                scale, shift = res.x
                z_opt = sorted([max(0.1, (bz * scale) + shift) for bz in bessel_base])
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 10) for zi in z_opt)) == 12:
                    B_val = res.fun
                    if B_val < best_B:
                        best_B = B_val
                        best_z = z_opt
        except Exception:
            pass
    
    # If we found a good candidate, verify with exact evaluation
    if best_z is not None and time.time() < deadline - 30:
        try:
            B_exact = evaluate_candidate_exact(best_z)
            if B_exact is not None:
                best_B = B_exact
        except Exception:
            pass
    
    # Fallback: return Bessel zeros if nothing better found
    if best_z is None:
        return bessel_base
    
    return best_z
