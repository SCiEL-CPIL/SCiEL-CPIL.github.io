"""Refined uncertainty inequality: Laguerre double roots (m=12, n=1)."""
import time
import math
import numpy as np
from fractions import Fraction

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 10.0  # seconds before hard limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN

def construct():
    """Return 12 double-root locations z_1..z_12 (list of floats)."""
    # Strategy: geometric spacing with optimization
    # z_i = base * ratio^i, tuned to minimize B
    
    # Initial geometric spread - small values to keep B low
    # We want z_i small so the product (x-z_i)^2 doesn't push roots up
    
    # Try: z_i = a * r^i for i=0..11
    # Optimize a and r to minimize B
    
    best_z = None
    best_B = float('inf')
    
    # Parameter space: a in [0.1, 2.0], r in [1.01, 1.5]
    # Coarse grid first
    a_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    r_vals = [1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for a in a_vals:
        if _time_remaining() < 30:
            break
        for r in r_vals:
            if _time_remaining() < 30:
                break
            z = [a * (r ** i) for i in range(12)]
            # Check constraints: finite, > 0, pairwise distinct
            if any(not math.isfinite(zi) or zi <= 0 for zi in z):
                continue
            if len(set(z)) != 12:
                continue
            # Quick numerical estimate of B (avoid full sympy for search)
            B_est = _estimate_B_fast(z)
            if B_est < best_B:
                best_B = B_est
                best_z = z
    
    # Refine around best found
    if best_z is not None:
        a0 = best_z[0]
        r0 = best_z[1] / best_z[0] if best_z[0] > 0 else 1.1
        
        # Local refinement
        for da in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            if _time_remaining() < 30:
                break
            for dr in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                if _time_remaining() < 30:
                    break
                a = a0 + da
                r = r0 + dr
                if a <= 0 or r <= 1.0:
                    continue
                z = [a * (r ** i) for i in range(12)]
                if any(not math.isfinite(zi) or zi <= 0 for zi in z):
                    continue
                if len(set(z)) != 12:
                    continue
                B_est = _estimate_B_fast(z)
                if B_est < best_B:
                    best_B = B_est
                    best_z = z
    
    if best_z is None:
        # Fallback
        best_z = [0.5 * (1.1 ** i) for i in range(12)]
    
    # Final: ensure sorted, positive, distinct
    best_z = sorted([float(zi) for zi in best_z])
    return best_z

def _estimate_B_fast(z):
    """
    Fast numerical estimate of B (largest positive root of q(x)).
    Uses numpy for speed; the exact sympy check happens in the evaluator.
    """
    n = len(z)
    # Build the system numerically to get g(x) coefficients
    # g(x) = sum_k c_k L_{2k}^{(-1/2)}(x), k=0..25
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=g'(z_i)=0 for i=0..11
    # Total: 2 + 24 = 26 equations, 26 unknowns (c_0..c_25)
    
    # We need L_{2k}^{(-1/2)}(x) and its derivative at x=0 and x=z_i
    # Use scipy or manual computation
    
    try:
        from scipy.special import genlaguerre
    except ImportError:
        return float('inf')
    
    alpha = -0.5
    degrees = [2*k for k in range(26)]  # 0, 2, 4, ..., 50
    
    # Evaluate Laguerre polynomials and derivatives at points
    points = [0.0] + list(z)
    
    # Build linear system A c = b
    # Row 0: g(0) = 0
    # Row 1: g'(0) = 1
    # Rows 2-25: g(z_i) = 0, g'(z_i) = 0
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Precompute Laguerre values and derivatives
    lag_vals = {}  # (deg, x) -> value
    lag_dvals = {}  # (deg, x) -> derivative
    
    for x in points:
        for deg in degrees:
            # Use scipy's genlaguerre
            try:
                L = genlaguerre(deg, alpha)
                val = float(L(x))
                dval = float(L.derivative()(x))
            except:
                return float('inf')
            lag_vals[(deg, x)] = val
            lag_dvals[(deg, x)] = dval
    
    # Row 0: g(0) = sum c_k L_{2k}(0) = 0
    for k, deg in enumerate(degrees):
        A[0, k] = lag_vals[(deg, 0.0)]
    b[0] = 0.0
    
    # Row 1: g'(0) = sum c_k L'_{2k}(0) = 1
    for k, deg in enumerate(degrees):
        A[1, k] = lag_dvals[(deg, 0.0)]
    b[1] = 1.0
    
    # Rows 2-25: for each z_i, g(z_i)=0 and g'(z_i)=0
    for i, zi in enumerate(z):
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        for k, deg in enumerate(degrees):
            A[row_g, k] = lag_vals[(deg, zi)]
            A[row_gp, k] = lag_dvals[(deg, zi)]
        b[row_g] = 0.0
        b[row_gp] = 0.0
    
    # Solve
    try:
        c = np.linalg.solve(A, b)
    except:
        return float('inf')
    
    # Now compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
    # Find roots of q(x) numerically
    # g(x) is a polynomial of degree 50 (from L_50)
    # Denominator: x * prod (x-z_i)^2 has degree 1 + 24 = 25
    # So q(x) is a polynomial of degree 50 - 25 = 25
    
    # Build g(x) as polynomial coefficients
    # L_n^{(alpha)}(x) = sum_{j=0}^n (-1)^j * binom(n+alpha, n-j) * x^j / j!
    
    def laguerre_coeffs(deg, alpha):
        """Return coefficients of L_deg^{(alpha)}(x) in ascending power order."""
        coeffs = np.zeros(deg + 1)
        for j in range(deg + 1):
            # (-1)^j * binom(deg+alpha, deg-j) / j!
            from math import lgamma, exp
            log_binom = lgamma(deg + alpha + 1) - lgamma(deg - j + 1) - lgamma(j + alpha + 1)
            coeff = ((-1) ** j) * exp(log_binom) / math.factorial(j)
            coeffs[j] = coeff
        return coeffs
    
    g_coeffs = np.zeros(51)  # degree up to 50
    for k, deg in enumerate(degrees):
        lc = laguerre_coeffs(deg, alpha)
        g_coeffs[:deg+1] += c[k] * lc
    
    # Denominator: x * prod_i (x - z_i)^2
    # Build as polynomial
    denom_coeffs = np.array([1.0])  # start with 1
    # Multiply by x
    denom_coeffs = np.concatenate([np.zeros(1), denom_coeffs])
    # Multiply by (x - z_i)^2 for each i
    for zi in z:
        # (x - zi)^2 = x^2 - 2*zi*x + zi^2
        factor = np.array([zi*zi, -2*zi, 1.0])  # ascending: [zi^2, -2zi, 1]
        denom_coeffs = np.convolve(denom_coeffs, factor)
    
    # Now divide g by denom to get q
    # Polynomial long division
    if len(denom_coeffs) > len(g_coeffs):
        return float('inf')
    
    # Pad g_coeffs
    g_pad = np.zeros(len(denom_coeffs) + len(g_coeffs) - 1)
    g_pad[len(g_pad)-len(g_coeffs):] = g_coeffs
    
    # Long division
    q_coeffs = np.zeros(len(g_pad) - len(denom_coeffs) + 1)
    rem = g_pad.copy()
    denom_lead = denom_coeffs[-1]
    
    for i in range(len(q_coeffs)):
        if i >= len(rem):
            break
        coeff = rem[len(rem) - 1 - i] / denom_lead
        q_coeffs[len(q_coeffs) - 1 - i] = coeff
        # Subtract coeff * denom * x^i
        denom_shifted = np.zeros(len(rem))
        denom_shifted[len(denom_shifted)-len(denom_coeffs):] = denom_coeffs
        rem[len(rem)-1-i:len(rem)-1-i+len(denom_coeffs)] -= coeff * denom_shifted[len(denom_shifted)-len(denom_coeffs):]
    
    # Find positive real roots of q
    try:
        roots = np.roots(q_coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        if real_roots:
            return max(real_roots)
        else:
            return 0.0
    except:
        return float('inf')
