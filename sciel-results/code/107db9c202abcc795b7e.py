import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a rational function structure
    with temperature-dependent coefficients.
    
    Structure:
    sigma = N(eps, T) / D(eps, T)
    
    Where:
    N = (p0 + p1*T + p2*T^2)*eps + (p3 + p4*T + p5*T^2)*eps^2
    D = 1 + (p6 + p7*T + p8*T^2)*eps + (p9)*eps^2
    
    This rational form can capture:
    - Linear elastic regime at small strain
    - Smooth transition to plastic behavior
    - Asymptotic hardening at large strain
    - Temperature-dependent softening/stiffening
    
    The model is linear in parameters for robust BFGS convergence.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize temperature to [0, 1] range for better conditioning
    t_min = 0.06667
    t_range = 0.93333
    t_norm = (t - t_min) / t_range
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Temperature-dependent coefficients for numerator
    a0 = p[0] + p[1] * t_norm + p[2] * t_norm ** 2
    a1 = p[3] + p[4] * t_norm + p[5] * t_norm ** 2
    
    # Temperature-dependent coefficients for denominator
    b1 = p[6] + p[7] * t_norm + p[8] * t_norm ** 2
    b2 = p[9]
    
    # Numerator: captures the stress behavior
    num = a0 * s + a1 * s ** 2
    
    # Denominator: provides rational function shape
    den = 1.0 + b1 * s + b2 * s ** 2
    
    # Prevent division by zero or very small numbers
    # Use a safe denominator that preserves sign
    abs_den = np.abs(den)
    sign_den = np.where(den >= 0, 1.0, -1.0)
    den_safe = np.where(abs_den < 1e-8, sign_den * 1e-8, den)
    
    stress = num / den_safe
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
