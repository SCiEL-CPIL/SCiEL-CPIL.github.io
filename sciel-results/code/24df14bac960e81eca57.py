"""Stress-strain model for an aluminium rod using Ramberg-Osgood formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood type formulation:
        sigma = E(t) * eps + K(t) * (eps_plastic)^n(t)
    
    where:
        E(t) = p[0] + p[1]*t + p[2]*t^2  (elastic modulus, polynomial in temp)
        K(t) = p[3] + p[4]*t + p[5]*t^2  (hardening coefficient)
        n(t) = p[6] + p[7]*t             (hardening exponent)
        p[8] = offset for elastic-plastic transition
        p[9] = additional temperature-dependent offset
    
    The plastic strain is defined as max(0, strain - strain_yield) to capture
    the elastic-plastic transition naturally.
    
    All 10 parameters are used, and the model is smooth (using softplus-like
    or clipped terms) to ensure BFGS can optimize effectively.
    """
    s = strain
    t = temp
    p = params
    
    # Temperature polynomials for material properties
    t2 = t * t
    
    # Elastic modulus: E(t) = p0 + p1*t + p2*t^2
    E = p[0] + p[1] * t + p[2] * t2
    
    # Hardening coefficient: K(t) = p3 + p4*t + p5*t^2
    K = p[3] + p[4] * t + p[5] * t2
    
    # Hardening exponent: n(t) = p6 + p7*t
    n = p[6] + p[7] * t
    
    # Yield strain offset: controls where plastic deformation begins
    eps_y = p[8]
    
    # Temperature-dependent offset
    offset = p[9] * t
    
    # Elastic strain component
    sigma_elastic = E * s
    
    # Plastic strain: only active beyond yield
    # Use a smooth approximation of max(0, s - eps_y) for differentiability
    eps_plastic = s - eps_y
    
    # Smooth positive part: use softplus-like or tanh-based smoothing
    # This ensures the function is differentiable everywhere for BFGS
    # Using: x_pos = (x + softplus(x)) / 2 where softplus(x) = log(1+exp(x))
    # But for numerical stability, use a simpler smooth max
    scale = 50.0  # sharpness of the transition
    eps_plastic_smooth = (eps_plastic + np.logaddexp(0, scale * eps_plastic) / scale) / 2.0
    
    # Ensure plastic strain is non-negative
    eps_plastic_smooth = np.maximum(eps_plastic_smooth, 0.0)
    
    # Hardening term: K * (eps_plastic)^n
    # Handle edge case where eps_plastic is 0 and n < 1
    # Use a small epsilon to avoid 0^negative issues
    eps_safe = np.maximum(eps_plastic_smooth, 1e-12)
    
    # Compute eps^n safely
    # For numerical stability, compute log then exp
    log_eps = np.log(eps_safe)
    log_hardening = np.log(np.maximum(K, 1e-12)) + n * log_eps
    sigma_plastic = np.exp(log_hardening)
    
    # Combine: total stress = elastic + plastic + offset
    sigma = sigma_elastic + sigma_plastic + offset
    
    # Clip to reasonable range to avoid numerical issues
    sigma = np.clip(sigma, -1e6, 1e6)
    
    return sigma
# EVOLVE-BLOCK-END
