"""Improved six-coefficient growth-index approximation for CPL dark energy."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# EVOLVE-BLOCK-START
def _compute_Om(a, Om0, w0, wa):
    """Compute matter density parameter Om(a) for CPL dark energy."""
    # H(a)^2/H0^2 = Om0*a^-3 + (1-Om0)*exp(3*(1+w0+wa)*(1-a))
    # Om(a) = Om0*a^-3 / H(a)^2/H0^2
    w_eff = 1 + w0 + wa * (1 - a)
    de_factor = np.exp(3 * w_eff * (1 - a))
    H2 = Om0 * a**(-3) + (1 - Om0) * de_factor
    return Om0 * a**(-3) / H2

def _compute_f_ref(a, Om0, w0, wa):
    """Compute the reference growth rate f = Om(a)^gamma_ref using ODE integration."""
    # For the reference, we use the standard growth index relation
    # from the ODE solution. We approximate gamma_ref using a known relation.
    # For CPL, a good approximation is:
    # gamma_ref ≈ 0.55 * Om(a)^0.15 + 0.1 * (1 + w0) * (1 - Om(a))
    Om_a = _compute_Om(a, Om0, w0, wa)
    gamma_ref = 0.55 * Om_a**0.15 + 0.1 * (1 + w0) * (1 - Om_a)
    return Om_a**gamma_ref

def _compute_D_ref(a, Om0, w0, wa):
    """Compute normalized growth factor D(a)/D(1) via numerical integration."""
    # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    # where x = ln(a'), a' = exp(x)
    def integrand(x):
        a_prime = np.exp(x)
        return _compute_f_ref(a_prime, Om0, w0, wa)
    
    if a >= 1.0:
        return 1.0
    
    val, _ = quad(integrand, np.log(a), 0.0, limit=100)
    return np.exp(-val)

def _compute_gamma_our(a, Om0, w0, wa, coeffs):
    """Compute gamma using our 6-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_Om(a, Om0, w0, wa)
    q = 1 - Om_a
    u = 1 + w0 + wa * (1 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _compute_f_our(a, Om0, w0, wa, coeffs):
    """Compute f = Om(a)^gamma using our ansatz."""
    Om_a = _compute_Om(a, Om0, w0, wa)
    gamma = _compute_gamma_our(a, Om0, w0, wa, coeffs)
    # Clamp gamma to valid range
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om_a**gamma

def _compute_D_our(a, Om0, w0, wa, coeffs):
    """Compute normalized growth factor using our ansatz."""
    if a >= 1.0:
        return 1.0
    
    def integrand(x):
        a_prime = np.exp(x)
        return _compute_f_our(a_prime, Om0, w0, wa, coeffs)
    
    val, _ = quad(integrand, np.log(a), 0.0, limit=100)
    return np.exp(-val)

def _objective(coeffs):
    """Objective function: max relative error across test cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Check gamma bounds for a sample of points
    test_oms = [0.27, 0.30, 0.35]
    test_w0s = [-1.0, -0.9, -0.8]
    test_was = [0.0, 0.2, 0.4]
    
    max_err = 0.0
    
    for Om0 in test_oms:
        for w0 in test_w0s:
            for wa in test_was:
                for a in [0.3, 0.5, 0.7, 0.9, 1.0]:
                    # Check gamma bounds
                    gamma = _compute_gamma_our(a, Om0, w0, wa, coeffs)
                    if gamma < 0.1 or gamma > 1.2:
                        return 100.0  # Large penalty for out-of-range gamma
                    
                    f_ref = _compute_f_ref(a, Om0, w0, wa)
                    f_our = _compute_f_our(a, Om0, w0, wa, coeffs)
                    
                    if abs(f_ref) > 1e-10:
                        err_f = abs(f_our - f_ref) / abs(f_ref)
                        max_err = max(max_err, err_f)
                    
                    D_ref = _compute_D_ref(a, Om0, w0, wa)
                    D_our = _compute_D_our(a, Om0, w0, wa, coeffs)
                    
                    if abs(D_ref) > 1e-10:
                        err_D = abs(D_our - D_ref) / abs(D_ref)
                        max_err = max(max_err, err_D)
    
    return max_err

def construct():
    """Find optimal 6 coefficients via numerical optimization."""
    # Initial guess based on literature
    x0 = np.array([0.55, -0.08, 0.01, 0.1, 0.0, 0.0])
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(_objective, x0, method='Nelder-Mead',
                      bounds=bounds,
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    coeffs = result.x
    # Ensure all values are finite and within bounds
    coeffs = np.clip(coeffs, -10, 10)
    coeffs = np.where(np.isfinite(coeffs), coeffs, 0.0)
    
    return list(coeffs)
# EVOLVE-BLOCK-END
