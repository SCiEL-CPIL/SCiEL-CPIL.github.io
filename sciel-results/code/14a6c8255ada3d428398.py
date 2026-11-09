"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from sympy import Rational, Matrix, Poly, real_roots, pi as sympy_pi, binomial, symbols, factor

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _L_val(n, alpha, z):
    """Compute L_n^(alpha)(z) exactly for rational z and alpha."""
    # L_n^(alpha)(z) = sum_{k=0}^n binom(n+alpha, n-k) (-z)^k / k!
    # Note: binom(n+alpha, n-k) is 0 if n-k < 0 or n+alpha < n-k (for non-integer alpha, this is never 0 in the same way, but the formula holds)
    # Actually, for generalized binomial, it's defined for all.
    # But for integer n, the sum terminates.
    result = Rational(0)
    for k in range(n + 1):
        # term = binom(n+alpha, n-k) * (-z)**k / factorial(k)
        # We need factorial(k)
        fk = math.factorial(k)
        term = binomial(n + alpha, n - k) * ((-z) ** k) / fk
        result += term
    return result

def _L_deriv_val(n, alpha, z):
    """Compute L_n^(alpha)'(z) = -L_{n-1}^(alpha+1)(z)."""
    if n == 0:
        return Rational(0)
    return -_L_val(n - 1, alpha + 1, z)

def _get_laguerre_coeffs(n, alpha):
    """Return list of coefficients [c_0, c_1, ..., c_n] for L_n^(alpha)(x) = sum c_i x^i."""
    coeffs = [Rational(0)] * (n + 1)
    for k in range(n + 1):
        # Coefficient of x^k in L_n^(alpha)(x) is binom(n+alpha, n-k) * (-1)^k / k!
        fk = math.factorial(k)
        coeffs[k] = binomial(n + alpha, n - k) * ((-1) ** k) / fk
    return coeffs

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Pre-compute Laguerre polynomial coefficients in monomial basis
    # We need L_0, L_2, ..., L_50
    # coeffs[k] is list of coeffs for L_{2k}
    lag_coeffs = []
    for k in range(26):
        n = 2 * k
        lag_coeffs.append(_get_laguerre_coeffs(n, alpha))
    
    # Build matrix A and vector b
    # Variables c_0 ... c_25
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Points to evaluate: 0, z_1, ..., z_12
    # We need values and derivatives at these points.
    # To speed up, we pre-compute L_{2k}(p) and L'_{2k}(p) for all p in points and all k.
    points = [Rational(0)] + z_rationals
    
    # Pre-compute values: vals[k][i] = L_{2k}(points[i])
    # Pre-compute derivs: derivs[k][i] = L'_{2k}(points[i])
    vals = [[0]*len(points) for _ in range(26)]
    derivs = [[0]*len(points) for _ in range(26)]
    
    for k in range(26):
        n = 2 * k
        for i, p in enumerate(points):
            vals[k][i] = _L_val(n, alpha, p)
            derivs[k][i] = _L_deriv_val(n, alpha, p)
            
    # Row 0: g(0) = 0
    # g(0) = sum c_k L_{2k}(0)
    for k in range(26):
        A[0, k] = vals[k][0]
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    # g'(0) = sum c_k L'_{2k}(0)
    for k in range(26):
        A[1, k] = derivs[k][0]
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i (index i-1 in z_rationals, index i in points)
    for i in range(12):
        point_idx = i + 1
        # g(z_i) = 0
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = vals[k][point_idx]
        b[row_idx, 0] = 0
        
        # g'(z_i) = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = derivs[k][point_idx]
        b[row_idx, 0] = 0
        
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None  # singular system
    
    # Construct g(x) coefficients in monomial basis
    # g(x) = sum_{k=0}^{25} c_k * sum_{j=0}^{2k} lag_coeffs[k][j] x^j
    # g(x) = sum_{j=0}^{50} G_j x^j
    max_deg = 50
    G = [Rational(0)] * (max_deg + 1)
    for k in range(26):
        ck = c[k]
        coeffs_k = lag_coeffs[k]
        for j in range(len(coeffs_k)):
            G[j] += ck * coeffs_k[j]
            
    # Construct denominator D(x) = x * prod (x - z_i)^2
    # D(x) = x * (x^2 - 2 z_i x + z_i^2) ...
    # We can build this polynomial coefficients directly.
    # Start with [1] for x
    D = [Rational(0)] * (2 * 12 + 1)
    D[1] = 1 # x
    for zi in z_rationals:
        # Multiply by (x - zi)^2 = x^2 - 2 zi x + zi^2
        new_D = [Rational(0)] * (len(D) + 2)
        for i in range(len(D)):
            new_D[i] += D[i] * (zi**2)
            new_D[i+1] += D[i] * (-2 * zi)
            new_D[i+2] += D[i] * 1
        D = new_D
        
    # Polynomial division G / D
    # G has degree 50, D has degree 25.
    # We want q(x) such that G = q * D + R.
    # Since the problem implies exact division (or we check remainder), we perform long division.
    # If remainder is not 0, the system might be inconsistent or my assumption is wrong?
    # The problem says "q(x) = g(x) / (x * prod_i (x - z_i)^2), an exact polynomial division."
    # This implies the remainder should be 0.
    
    q_coeffs = [Rational(0)] * (max_deg - len(D) + 1)
    rem = G[:]
    
    for i in range(len(q_coeffs)):
        # Determine leading term of rem
        # Find first non-zero coeff in rem
        lead_idx = None
        for j in range(len(rem)):
            if rem[j] != 0:
                lead_idx = j
                break
        if lead_idx is None:
            break
            
        # Leading coeff of rem is rem[lead_idx]
        # Leading coeff of D is D[-1] (which is 1)
        # Wait, D is stored [d_0, ..., d_deg]. Leading is D[-1].
        # But D is monic?
        # D(x) = x * prod (x-zi)^2. Leading term is x * x^{24} = x^{25}. Coeff 1.
        # So D[-1] should be 1.
        if D[-1] == 0:
            return None
            
        coeff = rem[lead_idx] / D[-1]
        q_coeffs[i] = coeff
        
        # Subtract coeff * x^i * D from rem
        for j in range(len(D)):
            rem[lead_idx + j] -= coeff * D[j]
            
    # Check remainder
    for val in rem:
        if val != 0:
            # Not exact division?
            # The problem statement implies it is. If not, maybe numerical error in my construction?
            # Or maybe the system is singular and I returned None earlier?
            # If we reach here, it's a problem.
            pass
            
    # Find real roots of q
    q_poly = Poly(sum(q_coeffs[i] * x**i for i in range(len(q_coeffs))), x)
    try:
        real_roots_list = real_roots(q_poly)
    except Exception:
        return None
        
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_poly.eval(r - eps)
            q_at_plus = q_poly.eval(r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _time_remaining(deadline):
    return deadline - time.time()

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    deadline = time.time() + 570.0  # 630s kill, keep 60s margin for safety
    
    # Compute Bessel zeros of J_0 for k=1..12
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Ensure positive, finite, distinct
    result = list(z_bessel)
    for i in range(12):
        if not math.isfinite(result[i]) or result[i] <= 0:
            result = [4.0 * (i + 1) for i in range(12)]
            break
    
    # Check pairwise distinct
    for i in range(12):
        for j in range(i+1, 12):
            if abs(result[i] - result[j]) < 1e-12:
                result = [4.0 * (i + 1) for i in range(12)]
                break
    
    # If time permits, try to refine with small perturbations
    remaining = _time_remaining(deadline)
    if remaining > 120:  # Need at least ~120s for one evaluation
        try:
            best_score = _compute_score_exact(result)
            if best_score is not None:
                # Try small perturbations of each z_i
                for i in range(12):
                    if _time_remaining(deadline) < 90:
                        break
                    for delta_frac in [0.01, -0.01, 0.02, -0.02]:
                        if _time_remaining(deadline) < 90:
                            break
                        candidate = list(result)
                        candidate[i] = result[i] * (1 + delta_frac)
                        if candidate[i] <= 0:
                            continue
                        # Check distinctness
                        valid = True
                        for a in range(12):
                            for b_idx in range(a+1, 12):
                                if abs(candidate[a] - candidate[b_idx]) < 1e-10:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        score = _compute_score_exact(candidate)
                        if score is not None and score < best_score:
                            best_score = score
                            result = candidate
        except Exception:
            pass
    
    return result
# EVOLVE-BLOCK-END
