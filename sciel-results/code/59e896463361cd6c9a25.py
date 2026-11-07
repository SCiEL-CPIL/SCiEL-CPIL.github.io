"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood based stress-strain model.
    sigma = E(t) * eps + K(t) * eps^n
    where E(t) and K(t) are low-order polynomials in temperature.
    n is fixed to a physically reasonable value for aluminium (e.g., 2.0) or fitted as p[9].
    To keep the problem well-conditioned for BFGS, we use a fixed exponent n=2.0 
    and fit the coefficients of E(t) and K(t).
    
    E(t) = p[0] + p[1]*t
    K(t) = p[2] + p[3]*t
    
    Wait, the directive says "parameterized as low-order polynomials in temperature".
    Let's use:
    E(t) = p[0] + p[1]*t + p[2]*t^2
    K(t) = p[3] + p[4]*t + p[5]*t^2
    n = p[6] (fitted exponent, but this makes it non-linear in params)
    
    Actually, fitting n with BFGS is fine, but it might be harder.
    Let's try to keep it simpler to ensure convergence.
    
    Alternative:
    sigma = (p[0] + p[1]*t) * s + (p[2] + p[3]*t) * s**p[4]
    
    This uses 5 parameters. We have 10.
    Let's expand the polynomials:
    E(t) = p[0] + p[1]*t + p[2]*t^2
    K(t) = p[3] + p[4]*t + p[5]*t^2
    n = p[6]
    
    sigma = E(t)*s + K(t)*s^n
    
    To handle s<0 (if any, though strain range is -0.0019 to 0.96), 
    s**n might be problematic if n is not integer. 
    Strain range is mostly positive. 
    Let's assume s >= 0 for the power term, or use abs(s) if needed?
    The training range is [-0.001911, 0.9636]. 
    If s is negative, s**n is undefined for non-integer n.
    Let's use s_pos = np.maximum(s, 0) for the plastic term?
    Or just s**n and hope the fitter avoids negative bases?
    Actually, BFGS might evaluate at negative s if it tries to fit the negative strain part.
    Let's use np.sign(s) * np.abs(s)**n for the hardening term?
    Ramberg-Osgood is usually for monotonic loading.
    
    Let's stick to:
    E(t) = p[0] + p[1]*t
    K(t) = p[2] + p[3]*t
    n = p[4]
    
    This is 5 params. We can add higher order temp terms.
    E(t) = p[0] + p[1]*t + p[2]*t^2
    K(t) = p[3] + p[4]*t + p[5]*t^2
    n = p[6]
    
    Let's use this.
    """
    s = strain
    t = temp
    p = params
    
    # Extract parameters
    E0, E1, E2 = p[0], p[1], p[2]
    K0, K1, K2 = p[3], p[4], p[5]
    n = p[6]
    
    # Calculate E(t) and K(t)
    E = E0 + E1 * t + E2 * t**2
    K = K0 + K1 * t + K2 * t**2
    
    # Handle the power term safely
    # If s is negative and n is non-integer, s**n is NaN.
    # The strain range includes small negative values.
    # Let's use np.where(s >= 0, s**n, 0) or similar?
    # Or use abs(s)**n * sign(s)?
    # For stress-strain, usually we care about magnitude.
    # Let's assume the data is mostly positive strain.
    # To be safe against NaNs during optimization:
    s_safe = np.maximum(s, 0)
    
    # Ramberg-Osgood: sigma = E*eps + K*eps^n
    # Note: This is linear in E and K, but non-linear in n.
    # BFGS should handle this.
    
    result = E * s + K * s_safe**n
    
    # Ensure finite values
    # If n is such that s_safe**n overflows or becomes inf, we might have issues.
    # But s is in [0, 1], so s**n is in [0, 1] for n > 0.
    # If n < 0, s**n can be inf at s=0.
    # Let's clamp n to a reasonable range? No, let the optimizer decide, but add a small epsilon to s_safe.
    s_safe = np.maximum(s, 1e-6)
    
    result = E * s + K * s_safe**n
    
    return result
# EVOLVE-BLOCK-END
