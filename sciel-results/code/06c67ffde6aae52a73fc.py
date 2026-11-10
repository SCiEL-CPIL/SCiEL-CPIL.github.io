"""Refined program for Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _laguerre_coeff_ratios(n, alpha_num, alpha_den):
    """
    Return coefficients of L_n^(alpha)(x) as list of Fractions [c_n, ..., c_0].
    alpha = alpha_num/alpha_den.
    Uses explicit formula: L_n^(alpha)(x) = sum_{k=0}^n C(n+alpha, n-k) (-1)^k x^k / k!
    where C(a,b) = Gamma(a+1)/(Gamma(b+1)*Gamma(a-b+1)) for generalized binomial.
    For our case alpha=-1/2, n is even (0,2,...,50).
    """
    if n == 0:
        return [Fraction(1)]
    
    # L_n^(alpha)(x) = sum_{k=0}^n binom(n+alpha, n-k) * (-x)^k / k!
    # binom(n+alpha, n-k) = (n+alpha)(n+alpha-1)...(n+alpha-(n-k)+1) / (n-k)!
    # = (n+alpha)(n+alpha-1)...(k+alpha) / (n-k)!
    # For alpha = -1/2, n even:
    # binom(n-1/2, n-k) = prod_{j=k}^{n-1} (j + 1/2) / (n-k)!  ... wait let me be careful
    
    # binom(a, b) for a = n + alpha, b = n - k
    # = Gamma(a+1) / (Gamma(b+1) * Gamma(a-b+1))
    # = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
    
    # For rational alpha, we can compute this exactly.
    # Let's use the recurrence or direct product.
    
    # Direct: binom(n+alpha, n-k) = prod_{j=0}^{n-k-1} (n+alpha-j) / (n-k)!
    # = prod_{j=0}^{n-k-1} (n + alpha - j) / (n-k)!
    
    # For alpha = -1/2: n + alpha - j = n - 1/2 - j = (2n-1-2j)/2
    # So binom(n-1/2, n-k) = prod_{j=0}^{n-k-1} (2n-1-2j)/2 / (n-k)!
    # = prod_{j=0}^{n-k-1} (2n-1-2j) / (2^(n-k) * (n-k)!)
    
    # The product prod_{j=0}^{m-1} (2n-1-2j) where m = n-k
    # = (2n-1)(2n-3)(2n-5)...(2n-1-2(m-1))
    # = (2n-1)(2n-3)...(2k+1)  [since 2n-1-2(n-k-1) = 2n-1-2n+2k+2 = 2k+1]
    # = (2n-1)!! / (2k-1)!!  where (-1)!! = 1
    
    # So binom(n-1/2, n-k) = (2n-1)!! / ((2k-1)!! * 2^(n-k) * (n-k)!)
    
    # And the coefficient of x^k is: binom(n-1/2, n-k) * (-1)^k / k!
    
    # Let's compute (2m-1)!! for m = 0, 1, 2, ...
    # (2m-1)!! = 1*3*5*...*(2m-1) = (2m)! / (2^m * m!)
    
    def double_factorial_odd(m):
        """(2m-1)!! = 1*3*5*...*(2m-1), with (-1)!! = 1"""
        if m <= 0:
            return Fraction(1)
        result = Fraction(1)
        for i in range(m):
            result *= Fraction(2*i + 1)
        return result
    
    coeffs = []
    for k in range(n + 1):
        # coeff of x^k = (-1)^k * binom(n-1/2, n-k) / k!
        m = n - k  # this is the "b" in binom(n+alpha, b)
        if m < 0:
            coeffs.append(Fraction(0))
            continue
        
        # binom(n-1/2, m) = (2n-1)!! / ((2m-1)!! * 2^m * m!)
        num = double_factorial_odd(n)
        den = double_factorial_odd(m) * (2 ** m) * math.factorial(m)
        binom_val = num / den
        
        sign = Fraction(-1) ** k
        coeff = sign * binom_val / math.factorial(k)
        coeffs.append(coeff)
    
    # coeffs[k] is coefficient of x^k, so return in order [c_0, c_1, ..., c_n]
    # But we want [c_n, c_{n-1}, ..., c_0] for numpy compatibility
    return list(reversed(coeffs))


def _laguerre_coeff_float(n, alpha):
    """Return coefficients of L_n^(alpha)(x) as numpy array [c_n, ..., c_0]."""
    if n == 0:
        return np.array([1.0])
    
    L_prev = np.array([1.0])
    L_curr = np.array([1.0 + alpha, -1.0])
    
    if n == 1:
        return L_curr
    
    for k in range(1, n):
        term1 = np.convolve(L_curr, [-1.0, 2.0 * k + alpha + 1.0])
        term2 = (k + alpha) * L_prev
        
        max_len = max(len(term1), len(term2))
        term1_padded = np.zeros(max_len)
        term2_padded = np.zeros(max_len)
        term1_padded[:len(term1)] = term1
        term2_padded[:len(term2)] = term2
        
        L_next = (term1_padded - term2_padded) / (k + 1.0)
        L_prev = L_curr
        L_curr = L_next
    
    return L_curr


def _compute_score_float(z_list):
    """Fast floating point approximation of the score."""
    z = np.array(z_list)
    if len(z) != 12:
        return 1e10
    if np.any(z <= 0):
        return 1e10
    if np.any(np.diff(np.sort(z)) < 1e-12):
        return 1e10
    
    alpha = -0.5
    points = np.concatenate(([0.0], z))
    N_pts = len(points)
    
    # Precompute Laguerre values at all points using recurrence
    L_minus = np.zeros((51, N_pts))
    L_plus = np.zeros((51, N_pts))
    
    L_minus[0, :] = 1.0
    L_plus[0, :] = 1.0
    L_minus[1, :] = 1.0 + alpha - points
    L_plus[1, :] = 1.0 - alpha - points  # Wait, L_1^(alpha)(x) = -x + 1 + alpha
    
    # Actually L_1^(alpha)(x) = -x + (1+alpha)
    # So L_1^(alpha)(x) at point p = -p + 1 + alpha
    L_minus[1, :] = -points + 1.0 + alpha
    L_plus[1, :] = -points + 1.0 - alpha  # This is for alpha=+1/2, but we only need alpha=-1/2
    
    # For derivatives: L_n'(x) can be computed via: d/dx L_n^(alpha)(x) = -L_{n-1}^{(alpha+1)}(x)
    # So L_n'(x) = -L_{n-1}^{(alpha+1)}(x)
    
    for k in range(1, 50):
        L_minus[k+1, :] = ((2.0 * k + alpha + 1.0 - points) * L_minus[k, :] - (k + alpha) * L_minus[k-1, :]) / (k + 1.0)
    
    # For derivatives, we need L_{n-1}^{(alpha+1)}(x) where alpha+1 = 1/2
    L_deriv = np.zeros((51, N_pts))
    L_deriv[0, :] = 0.0  # L_0' = 0
    L_deriv[1, :] = -1.0  # L_1' = -1
    
    # L_{n-1}^{(alpha+1)} for n=1..50, i.e., L_0^{(1/2)}, L_1^{(1/2)}, ..., L_{49}^{(1/2)}
    L_plus = np.zeros((50, N_pts))
    L_plus[0, :] = 1.0
    L_plus[1, :] = -points + 1.5  # L_1^(1/2)(x) = -x + 3/2
    
    for k in range(1, 49):
        L_plus[k+1, :] = ((2.0 * k + 1.5 - points) * L_plus[k, :] - (k + 0.5) * L_plus[k-1, :]) / (k + 1.0)
    
    # L_n'(x) = -L_{n-1}^{(alpha+1)}(x) for n >= 1
    for n in range(1, 51):
        L_deriv[n, :] = -L_plus[n-1, :]
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        n = 2 * k
        A[0, k] = L_minus[n, 0]
        A[1, k] = L_deriv[n, 0]
        
        for i in range(12):
            pt_idx = 1 + i
            A[2 + 2*i, k] = L_minus[n, pt_idx]
            A[3 + 2*i, k] = L_deriv[n, pt_idx]
            
    b[0] = 0.0
    b[1] = 1.0
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    # Build g(x) coefficients
    g_coeffs = np.zeros(51)
    for k in range(26):
        n = 2 * k
        L_coeffs = _laguerre_coeff_float(n, alpha)
        g_coeffs[:len(L_coeffs)] += c[k] * L_coeffs
    
    # Build D(x) = x * prod (x - z_i)^2
    D_coeffs = np.array([1.0, 0.0])
    for i in range(12):
        quad = np.array([1.0, -2.0 * z[i], z[i]**2])
        D_coeffs = np.convolve(D_coeffs, quad)
        
    try:
        q_coeffs, remainder = np.polydiv(g_coeffs, D_coeffs)
    except Exception:
        return 1e10
        
    roots = np.roots(q_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
            
    if not real_roots:
        return 1e10
        
    B = max(real_roots)
    if B < 0:
        B = 0.0
        
    score = B / (2.0 * math.pi)
    return score


def _compute_score_exact(z_list, deadline):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import (symbols, Rational, Poly, real_roots, pi as sympy_pi,
                       Matrix, binomial, factorial)
    
    if time.time() > deadline:
        return None
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build Laguerre polynomials using explicit formula with exact rationals
    laguerre_pols = []
    for d in range(0, 51, 2):
        # L_d^(alpha)(x) = sum_{k=0}^d binom(d+alpha, d-k) (-1)^k x^k / k!
        poly_terms = []
        for k in range(d + 1):
            # binom(d + alpha, d - k) where alpha = -1/2
            # = binom(d - 1/2, d - k)
            m = d - k
            if m < 0:
                continue
            # Compute binom(d - 1/2, m) exactly
            # = Gamma(d + 1/2) / (Gamma(m + 1) * Gamma(d - m + 1/2))
            # Use the product form:
            # binom(d-1/2, m) = prod_{j=0}^{m-1} (d - 1/2 - j) / m!
            # = prod_{j=0}^{m-1} (2d - 1 - 2j) / (2^m * m!)
            
            if m == 0:
                binom_val = Rational(1)
            else:
                num = Rational(1)
                for j in range(m):
                    num *= Rational(2 * d - 1 - 2 * j)
                den = Rational(2 ** m) * factorial(m)
                binom_val = num / den
            
            coeff = binom_val * Rational((-1) ** k) / factorial(k)
            if coeff != 0:
                poly_terms.append(coeff * x**k)
        
        laguerre_pols.append(sum(poly_terms) if poly_terms else Rational(0))
    
    if time.time() > deadline:
        return None
    
    # Build matrix A and vector b
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[0, 0] = 0
    b[1, 0] = 1
    
    for i in range(12):
        zi = z_rationals[i]
        row_val = 2 + 2 * i
        row_der = 3 + 2 * i
        for k in range(26):
            A[row_val, k] = laguerre_pols[k].subs(x, zi)
            A[row_der, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_val, 0] = 0
        b[row_der, 0] = 0
    
    if time.time() > deadline:
        return None
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        if time.time() > deadline:
            return None
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)


def _is_valid_z(z_list):
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


def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)


def construct():
    deadline = time.time() + 560.0
    
    # Try multiple initializations and pick the best
    initializations = []
    
    # Bessel zeros squared
    z_bessel = [_bessel_zero_J0(k)**2 for k in range(1, 13)]
    if _is_valid_z(z_bessel):
        initializations.append(z_bessel)
    
    # Geometric progression
    z_geo = [0.5 * (1.3 ** i) for i in range(12)]
    if _is_valid_z(z_geo):
        initializations.append(z_geo)
    
    # Linear spacing
    z_lin = [1.0 + i * 0.8 for i in range(12)]
    if _is_valid_z(z_lin):
        initializations.append(z_lin)
    
    # Quadratic spacing
    z_quad = [0.5 + 0.3 * i**1.5 for i in range(12)]
    if _is_valid_z(z_quad):
        initializations.append(z_quad)
    
    if not initializations:
        return [0.5 * (1.5 ** i) for i in range(12)]
    
    # Evaluate all initializations with float score
    best_z = None
    best_score_float = 1e10
    for z_init in initializations:
        s = _compute_score_float(z_init)
        if s < best_score_float:
            best_score_float = s
            best_z = list(z_init)
    
    if best_z is None:
        return [0.5 * (1.5 ** i) for i in range(12)]
    
    # Optimize using Nelder-Mead on log-space
    from scipy.optimize import minimize
    
    def objective(u):
        z = np.exp(u)
        if not _is_valid_z(z):
            return 1e10
        return _compute_score_float(z)
    
    u0 = np.log(np.array(best_z))
    
    # Pass 1: Nelder-Mead
    if time.time() < deadline - 180:
        try:
            res = minimize(objective, u0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.success:
                u_opt = res.x
                z_opt = np.exp(u_opt)
                if _is_valid_z(z_opt):
                    score_opt = _compute_score_float(z_opt)
                    if score_opt < best_score_float:
                        best_z = list(z_opt)
                        best_score_float = score_opt
        except Exception:
            pass
    
    # Pass 2: Coarse-to-fine local search
    if time.time() < deadline - 120:
        for scale in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() > deadline - 60:
                break
            improved = True
            while improved and time.time() < deadline - 60:
                improved = False
                for i in range(12):
                    if time.time() > deadline - 60:
                        break
                    for sign in [1, -1]:
                        if time.time() > deadline - 60:
                            break
                        cand = list(best_z)
                        cand[i] = best_z[i] * (1 + sign * scale)
                        if not _is_valid_z(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_score_float:
                            best_z = cand
                            best_score_float = s
                            improved = True
    
    # Pass 3: Pairwise perturbations
    if time.time() < deadline - 90:
        for i in range(12):
            if time.time() > deadline - 45:
                break
            for j in range(i + 1, 12):
                if time.time() > deadline - 45:
                    break
                for di in [0.02, -0.02, 0.01, -0.01]:
                    for dj in [0.02, -0.02, 0.01, -0.01]:
                        if time.time() > deadline - 45:
                            break
                        cand = list(best_z)
                        cand[i] = best_z[i] * (1 + di)
                        cand[j] = best_z[j] * (1 + dj)
                        if not _is_valid_z(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_score_float:
                            best_z = cand
                            best_score_float = s
    
    # Final polish
    if time.time() < deadline - 60:
        for _ in range(5):
            improved = False
            for i in range(12):
                if time.time() > deadline - 30:
                    break
                for scale in [0.001, 0.0005, 0.0002]:
                    for sign in [1, -1]:
                        if time.time() > deadline - 30:
                            break
                        cand = list(best_z)
                        cand[i] = best_z[i] * (1 + sign * scale)
                        if not _is_valid_z(cand):
                            continue
                        s = _compute_score_float(cand)
                        if s < best_score_float - 1e-15:
                            best_z = cand
                            best_score_float = s
                            improved = True
            if not improved:
                break
    
    # Exact evaluation
    final_score = None
    if time.time() < deadline - 120:
        final_score = _compute_score_exact(best_z, deadline - 30)
    
    # If time allows, try exact perturbations
    if time.time() < deadline - 150 and final_score is not None:
        for i in range(12):
            if time.time() > deadline - 60:
                break
            for step in [0.005, 0.002, 0.001]:
                for sign in [1, -1]:
                    if time.time() > deadline - 30:
                        break
                    cand = list(best_z)
                    cand[i] = best_z[i] * (1 + sign * step)
                    if not _is_valid_z(cand):
                        continue
                    try:
                        s = _compute_score_exact(cand, deadline - 15)
                        if s is not None and s < final_score:
                            final_score = s
                            best_z = cand
                    except Exception:
                        continue
    
    return best_z

# EVOLVE-BLOCK-END
