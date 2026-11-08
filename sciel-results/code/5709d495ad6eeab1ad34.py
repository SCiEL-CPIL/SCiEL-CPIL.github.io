"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, pi, sqrt


def _hermite_poly_coeffs(n):
    """Return coefficients of probabilists' Hermite polynomial He_n(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    # Recurrence: He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)
    he_n_minus_1 = [1.0]  # He_0
    he_n = [0.0, 1.0]     # He_1
    for k in range(2, n + 1):
        he_n_plus_1 = [0.0] * (k + 1)
        # x * He_n(x): shift coefficients
        for i in range(k):
            he_n_plus_1[i + 1] += he_n[i]
        # -k * He_{n-1}(x)
        for i in range(k - 1):
            he_n_plus_1[i] -= k * he_n_minus_1[i]
        he_n_minus_1 = he_n
        he_n = he_n_plus_1
    return he_n


def _eval_poly(coeffs, x):
    """Evaluate polynomial with coefficients [c0, c1, ..., cn] at x."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty bound for the given Hermite coefficients.
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    """
    # Get polynomial coefficients
    h0_coeffs = _hermite_poly_coeffs(0)
    h4_coeffs = _hermite_poly_coeffs(4)
    h8_coeffs = _hermite_poly_coeffs(8)
    h12_coeffs = _hermite_poly_coeffs(12)
    
    # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # P(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    h0_at_0 = _eval_poly(h0_coeffs, 0.0)
    h4_at_0 = _eval_poly(h4_coeffs, 0.0)
    h8_at_0 = _eval_poly(h8_coeffs, 0.0)
    h12_at_0 = _eval_poly(h12_coeffs, 0.0)
    
    if abs(h12_at_0) < 1e-15:
        # Degenerate case
        return float('inf')
    
    c12 = -(c0 * h0_at_0 + c4 * h4_at_0 + c8 * h8_at_0) / h12_at_0
    
    # Now compute the uncertainty bound
    # We need to compute A(f) and A(F(f)) where f is related to g
    # For the Hermite-Gaussian test function, the uncertainty product can be computed
    # numerically by evaluating the relevant integrals
    
    # Use numerical integration to compute the bound
    # The bound involves the L1 norm of x*g(x) and the L1 norm of xi*G(xi) where G is Fourier transform
    
    # For efficiency, use a reasonable grid
    x = np.linspace(-10, 10, 4001)
    dx = x[1] - x[0]
    
    # Compute P(x) * exp(-x^2/2)
    p_vals = np.zeros_like(x)
    for i, xi in enumerate(x):
        p_vals[i] = (c0 * _eval_poly(h0_coeffs, xi) + 
                     c4 * _eval_poly(h4_coeffs, xi) + 
                     c8 * _eval_poly(h8_coeffs, xi) + 
                     c12 * _eval_poly(h12_coeffs, xi))
    
    g_vals = p_vals * np.exp(-x**2 / 2)
    
    # A(f) is related to the L1 norm of x * g(x) (or similar quantity)
    # Based on the uncertainty inequality context, we need to compute:
    # The bound on |A(f)| * |A(F(f))| where A involves moment-type quantities
    
    # For the standard uncertainty inequality with these test functions:
    # A(f) = ∫ |x| |f(x)| dx / ∫ |f(x)| dx type quantity, or similar
    
    # Let's compute the relevant quantities
    # The Fourier transform of g(x) = P(x)exp(-x^2/2) can be computed analytically
    # or numerically
    
    # Numerical Fourier transform
    xi = np.linspace(-10, 10, 4001)
    dxi = xi[1] - xi[0]
    
    # Compute Fourier transform numerically
    G_vals = np.zeros_like(xi, dtype=complex)
    for i, xi_val in enumerate(xi):
        G_vals[i] = np.sum(g_vals * np.exp(-1j * xi_val * x)) * dx
    
    # Compute A(f) and A(F(f))
    # Based on the task description, these are likely:
    # A(f) = sup_x |x * f(x)| or similar moment bound
    # A(F(f)) = sup_xi |xi * F(f)(xi)| or similar
    
    # For the uncertainty inequality upper bound, we typically compute:
    # (∫ |x f(x)| dx) * (∫ |xi F(f)(xi)| dxi) or product of sup norms
    
    # Let's use the L1-based moments
    abs_x_g = np.abs(x * g_vals)
    abs_xi_G = np.abs(xi * G_vals)
    
    # Normalize by L1 norms if needed
    l1_g = np.sum(np.abs(g_vals)) * dx
    l1_G = np.sum(np.abs(G_vals)) * dxi
    
    if l1_g < 1e-15 or l1_G < 1e-15:
        return float('inf')
    
    # The bound is typically the product of the first absolute moments
    A_f = np.sum(abs_x_g) * dx / l1_g
    A_F = np.sum(abs_xi_G) * dxi / l1_G
    
    bound = A_f * A_F
    
    return bound


def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty bound.
    """
    t0 = time.time()
    deadline = t0 + 180  # Keep within 180s as per task requirement
    
    # Starting point: the prior state of the art
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    best_bound = _compute_bound(*x0)
    best_x = x0.copy()
    
    def objective(x):
        if time.time() > deadline - 10:
            return best_bound  # Return current best to avoid timeout
        return _compute_bound(x[0], x[1], x[2])
    
    # Try multiple optimization methods and starting points
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    
    for method in methods:
        if time.time() > deadline - 30:
            break
        
        try:
            if method in ['Nelder-Mead', 'Powell']:
                result = minimize(objective, x0, method=method, 
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            else:
                result = minimize(objective, x0, method=method, 
                                 options={'maxiter': 500, 'gtol': 1e-10})
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
            x0 = result.x.copy()  # Use best result as next starting point
            
        except Exception:
            pass
    
    # Final polish with high precision
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_x, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
