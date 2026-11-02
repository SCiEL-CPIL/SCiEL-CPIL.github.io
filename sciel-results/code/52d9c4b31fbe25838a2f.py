# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial

def _get_hermite_coeffs(n):
    """Returns coefficients of physicist's Hermite polynomial H_n(x) in ascending power order."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    c_nm1 = np.array([1.0])
    c_n = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        # H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)
        # 2x H_{k-1} shifts coeffs by 1
        term1 = np.zeros(len(c_n) + 1)
        term1[1:] = 2.0 * c_n
        # -2(k-1) H_{k-2}
        term2 = -2.0 * (k - 1) * c_nm1
        # Pad term2 to match length
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)))
        c_new = term1 + term2
        c_nm1 = c_n
        c_n = c_new
    return c_n

def _integrate_gaussian_weighted(coeffs, alpha):
    """
    Computes integral of exp(-alpha * x^2) * P(x) dx where P(x) has coeffs.
    Only even powers contribute.
    coeffs[i] is coeff of x^i.
    """
    integral = 0.0
    for i in range(0, len(coeffs), 2):
        c = coeffs[i]
        if c == 0:
            continue
        # Integral of x^i exp(-alpha x^2) from -inf to inf
        # Let u = sqrt(alpha) x, dx = du/sqrt(alpha)
        # x^i = u^i / alpha^(i/2)
        # Int = alpha^(-i/2) * alpha^(-0.5) * Int(u^i exp(-u^2))
        # Int(u^{2k} exp(-u^2)) = sqrt(pi) * (2k-1)!! / 2^k
        k = i // 2
        if k == 0:
            moment = np.sqrt(np.pi)
        else:
            # (2k-1)!! = (2k)! / (2^k k!)
            # Using log-gamma for stability or just factorial for small k
            # k is at most 24 (degree 48 / 2), so factorial is fine
            double_factorial = factorial(2 * k, exact=True) / (2 ** k * factorial(k, exact=True))
            moment = np.sqrt(np.pi) * double_factorial / (2 ** k)
        
        integral += c * (1.0 / alpha) ** (k + 0.5) * moment
    return integral

def _bound_analytic(params):
    a0, a1, a2 = params
    # Get coeffs for H0, H4, H8, H12
    h0 = _get_hermite_coeffs(0)
    h4 = _get_hermite_coeffs(4)
    h8 = _get_hermite_coeffs(8)
    h12 = _get_hermite_coeffs(12)
    
    # Construct P(x) = a0 H0 + a1 H4 + a2 H8 + a3 H12
    # Max degree is 12
    max_deg = 12
    p_coeffs = np.zeros(max_deg + 1)
    
    def add_poly(p, coeff, c):
        if c == 0: return
        # p is array of coeffs, c is scalar
        # align lengths
        if len(c) > len(p):
            p = np.pad(p, (0, len(c) - len(p)))
        p[:len(c)] += c * coeff
        
    # We need a3 such that P(0) = 0
    # P(0) = a0*H0(0) + a1*H4(0) + a2*H8(0) + a3*H12(0) = 0
    H0_0 = h0[0]
    H4_0 = h4[0]
    H8_0 = h8[0]
    H12_0 = h12[0]
    
    if H12_0 == 0:
        return 1e18 # Should not happen
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    
    # Build P
    p_coeffs = np.zeros(13)
    p_coeffs[:len(h0)] += a0 * h0
    p_coeffs[:len(h4)] += a1 * h4
    p_coeffs[:len(h8)] += a2 * h8
    p_coeffs[:len(h12)] += a3 * h12
    
    # Calculate I2 = int exp(-x^2) P(x)^2 dx
    p2 = np.convolve(p_coeffs, p_coeffs)
    I2 = _integrate_gaussian_weighted(p2, 1.0)
    
    if I2 <= 0:
        return 1e18
        
    # Calculate I4 = int exp(-2x^2) P(x)^4 dx
    p4 = np.convolve(p2, p2)
    I4 = _integrate_gaussian_weighted(p4, 2.0)
    
    # The bound is sqrt(I4) * sqrt(I4) / I2 = I4 / I2
    # Note: The seed code used sqrt(int_g4)*sqrt(int_G4)/int_g2.
    # Since int_G4 == int_g4 for this specific class of functions (Gaussian*Hermite),
    # the expression is I4 / I2.
    
    return I4 / I2

def construct():
    # Define bounds for a0, a1, a2
    # Literature seed: [-1.13, 0.04, 0.0003]
    # We can search in a reasonable range
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # Global search
    res_global = differential_evolution(
        _bound_analytic,
        bounds=bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=False,
        init='sobol'
    )
    
    best_params = res_global.x
    best_val = res_global.fun
    
    # Local refinement
    try:
        res_local = minimize(
            _bound_analytic,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res_local.fun < best_val:
            best_params = res_local.x
            best_val = res_local.fun
    except Exception:
        pass
        
    return np.asarray(best_params, dtype=float)
# EVOLVE-BLOCK-END
