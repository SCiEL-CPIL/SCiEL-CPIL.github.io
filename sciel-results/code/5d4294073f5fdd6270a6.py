import sympy as sp
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 570.0

    # Compute the 12 smallest positive zeros of L_12^(-1/2)(x)
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Get the generalized Laguerre polynomial L_12^(-1/2)(x)
    L12 = sp.Laguerre(12, alpha, x)
    
    # Find all roots
    try:
        roots = sp.roots(L12, x)
    except Exception:
        roots = {}
    
    # Extract positive real roots
    positive_roots = []
    for r, mult in roots.items():
        try:
            rf = float(r)
            if rf > 0 and np.isfinite(rf):
                positive_roots.append(rf)
        except Exception:
            continue
    
    # If we don't have exactly 12 positive roots, try numeric approach
    if len(positive_roots) < 12:
        # Use mpmath for numeric root finding
        import mpmath as mp
        mp.mp.dps = 50
        alpha_mp = mp.mpf(-1) / 2
        
        def L12_mp(xv):
            total = mp.mpf(0)
            for j in range(13):
                total += ((-1) ** j) * mp.binomial(12 + alpha_mp, 12 - j) * (xv ** j) / mp.factorial(j)
            return total
        
        # Scan for sign changes
        positive_roots = []
        prev_x = mp.mpf(0.001)
        prev_val = L12_mp(prev_x)
        step = mp.mpf(0.1)
        for i in range(1, 1000):
            cur_x = prev_x + step
            cur_val = L12_mp(cur_x)
            if prev_val * cur_val < 0:
                try:
                    root = mp.findroot(L12_mp, (prev_x, cur_x))
                    if root > 0:
                        positive_roots.append(float(root))
                except Exception:
                    pass
            prev_x = cur_x
            prev_val = cur_val
        
        positive_roots.sort()
    
    # Take the 12 smallest
    positive_roots.sort()
    z_laguerre = positive_roots[:12]
    
    # Validate
    if len(z_laguerre) != 12:
        # Fallback: use Bessel zero squares
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        z_laguerre = [z * z for z in j0_zeros]
    
    # Ensure pairwise distinct and > 0
    z_laguerre = sorted(z_laguerre)
    if any(zi <= 0 for zi in z_laguerre):
        j0_zeros = [
            2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
            14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
            27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
        ]
        z_laguerre = [z * z for z in j0_zeros]
    
    if len(set(round(zi, 10) for zi in z_laguerre)) != 12:
        # Add small perturbations to make distinct
        for i in range(len(z_laguerre)):
            z_laguerre[i] += 1e-10 * (i + 1)
    
    # Now try to optimize using the Laguerre nodes as starting point
    # Use a fast mpmath-based objective for local search
    
    def evaluate_fast(z_list):
        """Fast mpmath evaluation. Returns B estimate or large penalty."""
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

            def _gen_laguerre(n, alpha, x_val):
                total = mp.mpf(0)
                for j in range(n + 1):
                    total += ((-1) ** j) * mp.binomial(n + alpha, n - j) * (x_val ** j) / mp.factorial(j)
                return total

            def _gen_laguerre_deriv(n, alpha, x_val):
                if n == 0:
                    return mp.mpf(0)
                return -_gen_laguerre(n - 1, alpha + 1, x_val)

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

            for xi in xs:
                xv = mp.mpf(xi)
                try:
                    qi = q_val(xv)
                except Exception:
                    prev_q = None
                    continue
                if prev_q is not None and prev_q * qi < 0:
                    a_idx = list(xs).index(xi) - 1
                    if a_idx < 0:
                        a_idx = 0
                    a = mp.mpf(xs[a_idx])
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
                return 1e10
            return max(roots_found)
        except Exception:
            return 1e10

    def evaluate_exact(z_list):
        """Exact sympy evaluation. Returns B (float) or None if invalid."""
        try:
            z_rat = [sp.Rational(float(z)) for z in z_list]
            x = sp.Symbol('x')
            alpha = sp.Rational(-1, 2)

            Ls = [sp.Laguerre(2 * k, alpha, x) for k in range(26)]

            M = sp.zeros(26, 26)
            rhs = sp.zeros(26, 1)

            for k in range(26):
                M[0, k] = Ls[k].subs(x, 0)
            rhs[0] = 0

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

    # First, exact-evaluate the Laguerre node candidate
    best_B = float('inf')
    best_z = None
    
    if time.time() < deadline - 120:
        B_lag = evaluate_exact(z_laguerre)
        if B_lag is not None:
            best_B = B_lag
            best_z = z_laguerre[:]
    
    # Also try Bessel zero squares as alternative
    j0_zeros = [
        2.4048255577, 5.5200781103, 8.6537279129, 11.7915289410,
        14.9309177085, 18.0710639679, 21.2116366299, 24.3524715308,
        27.4934791320, 30.6346064684, 33.7758202136, 36.9170984358
    ]
    bessel_sq = [z * z for z in j0_zeros]
    
    if time.time() < deadline - 60:
        B_bessel = evaluate_exact(bessel_sq)
        if B_bessel is not None and (best_z is None or B_bessel < best_B):
            best_B = B_bessel
            best_z = bessel_sq[:]
    
    # Use fast evaluator to try local improvements from both starting points
    from scipy.optimize import minimize
    
    candidates = []
    if best_z is not None:
        candidates.append(best_z[:])
    candidates.append(z_laguerre[:])
    candidates.append(bessel_sq[:])
    
    for start in candidates:
        if time.time() > deadline - 90:
            break
        try:
            res = minimize(evaluate_fast, start, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res.success or res.fun < best_B:
                z_opt = sorted(res.x)
                if all(zi > 0 for zi in z_opt) and len(set(round(zi, 8) for zi in z_opt)) == 12:
                    if time.time() < deadline - 30:
                        B_exact = evaluate_exact(z_opt)
                        if B_exact is not None and B_exact < best_B:
                            best_B = B_exact
                            best_z = z_opt
        except Exception:
            pass
    
    if best_z is None:
        best_z = z_laguerre
    
    return best_z
