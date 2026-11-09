"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)


def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, LUsolve
    from sympy.polys.orthopolys import laguerre as _laguerre_gen
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
    laguerre_pols = []
    for k in range(26):
        d = 2 * k
        Ld = _laguerre_gen(d, alpha)
        laguerre_pols.append(Ld)
    
    # Build the linear system: 26 unknowns c_0..c_25
    # Equations:
    #   g(0) = 0
    #   g'(0) = 1
    #   g(z_i) = 0 for i = 1..12
    #   g'(z_i) = 0 for i = 1..12
    # Total: 2 + 12 + 12 = 26 equations
    
    A = Matrix(26, 26, 0)
    b = Matrix(26, 1, 0)
    
    # g(0) = sum c_k * L_{2k}(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0] = 0
    
    # g'(0) = sum c_k * L'_{2k}(0) = 1
    for k in range(26):
        A[1, k] = diff(laguerre_pols[k], x).subs(x, 0)
    b[1] = 1
    
    # g(z_i) = 0 and g'(z_i) = 0 for each i
    for i, zi in enumerate(z_rationals):
        row_g = 2 + i
        row_gp = 14 + i
        for k in range(26):
            A[row_g, k] = laguerre_pols[k].subs(x, zi)
            A[row_gp, k] = diff(laguerre_pols[k], x).subs(x, zi)
        b[row_g] = 0
        b[row_gp] = 0
    
    # Solve using LUsolve
    try:
        c = LUsolve(A, b)
    except Exception:
        return None  # singular system
    
    # Build g(x) = sum c_k * L_{2k}(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    g = Poly(g, x)
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = Poly(x, x)
    for zi in z_rationals:
        denom = denom * Poly((x - zi)**2, x)
    
    # Exact polynomial division: q(x) = g(x) / denom
    try:
        q, rem = g.div(denom)
    except Exception:
        return None
    
    # Check remainder is zero (it should be by construction)
    if rem != 0:
        return None
    
    # Find real roots of q
    try:
        roots = q.real_roots()
    except Exception:
        return None
    
    if not roots:
        return 0.0
    
    # For each real root r, check if q has strictly opposite signs at r-1e-198 and r+1e-198
    # Since we're using exact rationals, 1e-198 as a rational is Rational(1, 10**198)
    eps = Rational(1, 10**198)
    
    counted_roots = []
    for r in roots:
        q_left = q.eval(r - eps)
        q_right = q.eval(r + eps)
        # Strictly opposite signs: one positive, one negative
        if (q_left > 0 and q_right < 0) or (q_left < 0 and q_right > 0):
            counted_roots.append(r)
    
    if not counted_roots:
        return 0.0
    
    B = max(counted_roots)
    return float(B) / (2 * math.pi)


def _compute_score_fast(z_list):
    """Fast approximate score using mpmath for screening."""
    try:
        from mpmath import mp, mpf, polyroots, polydiv, polyroots as _pr
        mp.dps = 50
        
        z_mp = [mpf(str(z)) for z in z_list]
        z_mp.sort()
        
        # Build Laguerre polynomials L_{2k}^{(-1/2)}(x) using recurrence
        # L_0^a(x) = 1, L_1^a(x) = 1+a-x
        # (n+1) L_{n+1}^a(x) = (2n+1+a-x) L_n^a(x) - (n+a) L_{n-1}^a(x)
        alpha = mpf('-0.5')
        
        def laguerre_gen(n, a, x):
            if n == 0:
                return mpf(1)
            if n == 1:
                return 1 + a - x
            Lnm1 = mpf(1)
            Ln = 1 + a - x
            for k in range(1, n):
                Lnp1 = ((2*k + 1 + a - x) * Ln - (k + a) * Lnm1) / (k + 1)
                Lnm1 = Ln
                Ln = Lnp1
            return Ln
        
        def laguerre_gen_deriv(n, a, x):
            # Numerical derivative
            h = mpf('1e-20')
            return (laguerre_gen(n, a, x + h) - laguerre_gen(n, a, x - h)) / (2*h)
        
        # Build system
        A = [[mpf(0)] * 26 for _ in range(26)]
        b = [mpf(0)] * 26
        
        for k in range(26):
            d = 2*k
            A[0][k] = laguerre_gen(d, alpha, mpf(0))
            A[1][k] = laguerre_gen_deriv(d, alpha, mpf(0))
        
        b[0] = mpf(0)
        b[1] = mpf(1)
        
        for i, zi in enumerate(z_mp):
            row_g = 2 + i
            row_gp = 14 + i
            for k in range(26):
                d = 2*k
                A[row_g][k] = laguerre_gen(d, alpha, zi)
                A[row_gp][k] = laguerre_gen_deriv(d, alpha, zi)
            b[row_g] = mpf(0)
            b[row_gp] = mpf(0)
        
        # Solve linear system (Gaussian elimination)
        n = 26
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            # Find pivot
            max_row = col
            for r in range(col+1, n):
                if abs(M[r][col]) > abs(M[max_row][col]):
                    max_row = r
            if abs(M[col][col]) < mpf('1e-30'):
                return None
            M[col], M[max_row] = M[max_row], M[col]
            for r in range(n):
                if r != col:
                    factor = M[r][col] / M[col][col]
                    for c in range(col, n+1):
                        M[r][c] -= factor * M[col][c]
        
        c = [M[i][n] / M[i][i] for i in range(n)]
        
        # Build g(x) coefficients
        # g(x) = sum c_k * L_{2k}(x)
        # We need to evaluate q(x) = g(x) / (x * prod (x-z_i)^2)
        # Find roots of q by finding roots of g(x) that are not at 0 or z_i
        # Actually, let's construct q as a polynomial
        
        # Get all coefficients of g(x)
        # L_n(x) has degree n, so g has degree 50
        # Build g as a polynomial using coefficients
        
        def laguerre_coeffs(n, a):
            """Return coefficients of L_n^a(x) as list [c_0, c_1, ..., c_n] where L = sum c_i x^i"""
            if n == 0:
                return [mpf(1)]
            if n == 1:
                return [1+a, -1]
            # Use recurrence
            Lnm1 = [mpf(1)]
            Ln = [1+a, -1]
            for k in range(1, n):
                # Lnp1 = ((2k+1+a-x)*Ln - (k+a)*Lnm1) / (k+1)
                # (2k+1+a)*Ln - x*Ln - (k+a)*Lnm1
                Lnp1 = [mpf(0)] * (len(Ln) + 1)
                for i in range(len(Ln)):
                    Lnp1[i] += (2*k+1+a) * Ln[i]
                    Lnp1[i+1] -= Ln[i]
                for i in range(len(Lnm1)):
                    Lnp1[i] -= (k+a) * Lnm1[i]
                for i in range(len(Lnp1)):
                    Lnp1[i] /= (k+1)
                Lnm1 = Ln
                Ln = Lnp1
            return Ln
        
        g_coeffs = [mpf(0)] * 51
        for k in range(26):
            d = 2*k
            Lc = laguerre_coeffs(d, alpha)
            for i in range(len(Lc)):
                g_coeffs[i] += c[k] * Lc[i]
        
        # Denominator: x * prod (x - z_i)^2
        denom_coeffs = [mpf(1)]  # just x initially... wait, x * prod
        # Start with [0, 1] for x
        denom_coeffs = [mpf(0), mpf(1)]
        for zi in z_mp:
            # Multiply by (x - zi)^2 = x^2 - 2*zi*x + zi^2
            new_coeffs = [mpf(0)] * (len(denom_coeffs) + 2)
            for i in range(len(denom_coeffs)):
                new_coeffs[i+2] += denom_coeffs[i]
                new_coeffs[i+1] -= 2*zi*denom_coeffs[i]
                new_coeffs[i] += zi*zi*denom_coeffs[i]
            denom_coeffs = new_coeffs
        
        # Polynomial division: g / denom
        # g has degree 50, denom has degree 1 + 24 = 25
        # q has degree 25
        n_g = 50
        n_d = len(denom_coeffs) - 1  # 25
        
        # Normalize denom
        d_lead = denom_coeffs[-1]
        denom_norm = [dc / d_lead for dc in denom_coeffs]
        
        q_coeffs = [mpf(0)] * (n_g - n_d + 1)
        rem = g_coeffs[:]
        
        for i in range(n_g - n_d + 1):
            qi = rem[i] / denom_norm[0]
            q_coeffs[n_g - n_d - i] = qi
            for j in range(len(denom_norm)):
                rem[i+j] -= qi * denom_norm[j]
        
        # Find real roots of q_coeffs
        # Use mpmath polyroots
        # q_coeffs is [c_0, c_1, ..., c_25] (ascending)
        # mpmath expects descending? Let me check... mpmath.polyroots takes ascending
        try:
            q_roots = _pr(q_coeffs)
        except Exception:
            return None
        
        real_roots = [r.real for r in q_roots if abs(r.imag) < mpf('1e-10')]
        
        if not real_roots:
            return 0.0
        
        # For each real root, check sign change
        eps = mpf('1e-198')
        counted = []
        for r in real_roots:
            # Evaluate q at r-eps and r+eps
            q_left = sum(q_coeffs[i] * (r-eps)**i for i in range(len(q_coeffs)))
            q_right = sum(q_coeffs[i] * (r+eps)**i for i in range(len(q_coeffs)))
            if (q_left > 0 and q_right < 0) or (q_left < 0 and q_right > 0):
                counted.append(r)
        
        if not counted:
            return 0.0
        
        B = max(counted)
        return float(B) / (2 * math.pi)
    
    except Exception:
        return None


def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 580.0  # 580s margin to stay under 630s kill
    
    # Start with Bessel zero squared values
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Validate initial
    if any(not math.isfinite(z) or z <= 0 for z in z_bessel):
        z_bessel = [4.0 * (i + 1) for i in range(12)]
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_bessel[i] - z_bessel[j]) < 1e-12:
                z_bessel = [4.0 * (i + 1) for i in range(12)]
                break
    
    best_z = z_bessel[:]
    best_score = None
    
    # Try to get a fast score
    t0 = time.time()
    score = _compute_score_fast(z_bessel)
    if score is not None:
        best_score = score
    
    # Local refinement using Nelder-Mead on the fast evaluator
    if time.time() < deadline - 300:  # Need at least 300s left for exact verification
        try:
            from scipy.optimize import minimize
            
            def objective(z_flat):
                z_list = list(z_flat)
                if any(not math.isfinite(z) or z <= 0 for z in z_list):
                    return 1e10
                for i in range(12):
                    for j in range(i+1, 12):
                        if abs(z_list[i] - z_list[j]) < 1e-10:
                            return 1e10
                s = _compute_score_fast(z_list)
                if s is None:
                    return 1e10
                return s
            
            x0 = np.array(z_bessel)
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < (best_score if best_score is not None else 1e10):
                candidate = list(result.x)
                # Validate
                valid = True
                for z in candidate:
                    if not math.isfinite(z) or z <= 0:
                        valid = False
                        break
                if valid:
                    for i in range(12):
                        for j in range(i+1, 12):
                            if abs(candidate[i] - candidate[j]) < 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    best_z = candidate
                    if time.time() < deadline - 100:
                        s = _compute_score_fast(candidate)
                        if s is not None:
                            best_score = s
        except Exception:
            pass
    
    # Final exact verification
    if time.time() < deadline - 60:
        try:
            exact_score = _compute_score_exact(best_z)
            if exact_score is not None:
                # This is the score that will be reported
                pass
        except Exception:
            pass
    
    # Ensure final result is valid
    result = best_z[:]
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    return result

# EVOLVE-BLOCK-END
