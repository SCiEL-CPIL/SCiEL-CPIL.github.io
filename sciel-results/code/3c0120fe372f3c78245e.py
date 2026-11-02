"""Rewritten program: refined optimization for Hermite-Gaussian uncertainty bound."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import time
import math


def _hermite_coeffs(n):
    """Return coefficients of H_n(x) in ascending order (constant to x^n)."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    # H_{n}(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    h_n_minus_2 = _hermite_coeffs(n-2)
    h_n_minus_1 = _hermite_coeffs(n-1)
    
    # 2x * H_{n-1}(x)
    # Shift h_n_minus_1 by 1 and multiply by 2
    term1 = np.zeros(n)
    term1[1:] = 2.0 * h_n_minus_1
    
    # -2(n-1) * H_{n-2}(x)
    term2 = -2.0 * (n-1) * h_n_minus_2
    
    # Pad term2 to length n if necessary
    if len(term2) < n:
        term2 = np.pad(term2, (0, n - len(term2)))
        
    return term1 + term2


# Precompute coefficients for H0, H4, H8, H12
# Note: _hermite_coeffs returns [c0, c1, ..., cn]
H0_c = _hermite_coeffs(0)
H4_c = _hermite_coeffs(4)
H8_c = _hermite_coeffs(8)
H12_c = _hermite_coeffs(12)

# Pad to length 13 (degree 12) for easy combination
def _pad_coeff(c, length):
    if len(c) >= length:
        return c[:length]
    return np.pad(c, (0, length - len(c)))

H0_p = _pad_coeff(H0_c, 13)
H4_p = _pad_coeff(H4_c, 13)
H8_p = _pad_coeff(H8_c, 13)
H12_p = _pad_coeff(H12_c, 13)

# Coefficients of P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# We need to compute c12 from c0, c4, c8
# P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
# H_n(0) is the constant term of H_n(x)
H0_0 = H0_c[0]
H4_0 = H4_c[0]
H8_0 = H8_c[0]
H12_0 = H12_c[0]


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) for given coefficients.
    
    Uses exact Hermite-Gaussian Fourier properties.
    Since all degrees are multiples of 4, F(k) = sqrt(2*pi)*g(k),
    so the product is sqrt(2*pi) * (sup|x*g(x)|)^2.
    We find the supremum by evaluating h(x) = x*g(x) at the roots of h'(x).
    """
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Construct coefficients of P(x)
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Coeffs are in ascending order
    P_c = c0 * H0_p + c4 * H4_p + c8 * H8_p + c12 * H12_p
    
    # h(x) = x * P(x) * exp(-x^2/2)
    # h'(x) = (P(x) + x*P'(x) - x^2*P(x)) * exp(-x^2/2)
    # Let Q(x) = P(x) + x*P'(x) - x^2*P(x)
    # We need roots of Q(x).
    
    # P'(x) coefficients: if P_c = [p0, p1, ..., p12], 
    # P'_c[i] = (i+1)*p_{i+1} for i=0..11
    P_prime_c = np.zeros(12)
    for i in range(12):
        P_prime_c[i] = (i + 1) * P_c[i + 1]
        
    # Term 1: P(x)
    # Coeffs: P_c (length 13)
    
    # Term 2: x * P'(x)
    # Shift P_prime_c by 1
    x_P_prime_c = np.zeros(13)
    x_P_prime_c[1:] = P_prime_c
    
    # Term 3: -x^2 * P(x)
    # Shift P_c by 2 and multiply by -1
    neg_x2_P_c = np.zeros(13)
    neg_x2_P_c[2:] = -P_c[:-2]
    
    Q_c = P_c + x_P_prime_c + neg_x2_P_c
    
    # Find roots of Q(x)
    # Filter out very small coefficients to avoid numerical noise issues if any, 
    # though np.roots handles it reasonably.
    roots = np.roots(Q_c)
    
    # Keep only real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    
    if len(real_roots) == 0:
        # Fallback to grid if something went wrong, though unlikely
        x = np.linspace(-10, 10, 1000)
        P_vals = np.polyval(P_c[::-1], x) # np.polyval expects descending
        g_vals = P_vals * np.exp(-x**2 / 2.0)
        xg = x * g_vals
        Af = np.max(np.abs(xg))
    else:
        # Evaluate h(x) at real roots
        # h(x) = x * P(x) * exp(-x^2/2)
        P_vals = np.polyval(P_c[::-1], real_roots)
        g_vals = P_vals * np.exp(-real_roots**2 / 2.0)
        xg = real_roots * g_vals
        Af = np.max(np.abs(xg))
        
        # Also check behavior at infinity? 
        # h(x) -> 0 as x -> inf.
        # Check if P(x) has roots that might be missed? 
        # The critical points are the only candidates for local extrema.
        # Global max is among them.
        
    bound = math.sqrt(2 * math.pi) * Af * Af
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    try:
        return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
    except:
        return 1e10


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit
    
    # Initial guess
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_val = _objective(x0)
    best_x = x0.copy()
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(-3.0, 0.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=15,
                tol=1e-10, seed=42,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Local refinement with multiple starting points
    start_points = [
        best_x.copy(),
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.1, 0.04, 0.0003]),
        np.array([-1.15, 0.035, 0.00028]),
        np.array([-1.05, 0.045, 0.00035]),
        np.array([-1.2, 0.03, 0.00025]),
        np.array([-1.0, 0.05, 0.0004]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, sp, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
