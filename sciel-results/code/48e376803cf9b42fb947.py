import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod combining Ramberg-Osgood physics
    with a rational function denominator for smooth asymptotic behavior.
    
    Structure:
    sigma = [E(T)*eps + K(T)*|eps|^n + S(T)*sigmoid(eps)] / [1 + b1(T)*eps + b2(T)*eps^2]
    
    Where:
    - E(T) = p0 + p1*T  (elastic modulus, linear in T)
    - K(T) = p2 + p3*T  (hardening coefficient, linear in T)
    - n = 0.15 (fixed hardening exponent for aluminium)
    - S(T) = p4 + p5*T  (yield transition amplitude)
    - b1(T) = p6 + p7*T (denominator linear coeff)
    - b2(T) = p8 + p9*T (denominator quadratic coeff)
    
    The numerator captures:
    - Linear elastic regime at small strain
    - Power-law plastic hardening (Ramberg-Osgood)
    - Smooth yield transition via sigmoid
    
    The denominator provides:
    - Rational function shape for asymptotic hardening
    - Temperature-dependent curvature adjustments
    - Smooth behavior across the full strain range
    
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
    
    # Fixed physical constants
    # Hardening exponent for aluminium
    n = 0.15
    
    # Compute the nonlinear strain term for Ramberg-Osgood hardening
    eps_abs = np.abs(s)
    eps_safe = np.maximum(eps_abs, 1e-12)
    eps_power = np.power(eps_safe, n)
    
    # Sigmoid for smooth yield transition
    # Yield strain for aluminium is typically ~0.001-0.005
    yield_center = 0.003
    yield_width = 0.008
    sigmoid = 0.5 * (1.0 + np.tanh((s - yield_center) / yield_width))
    
    # Temperature-dependent coefficients
    E_T = p[0] + p[1] * t_norm
    K_T = p[2] + p[3] * t_norm
    S_T = p[4] + p[5] * t_norm
    b1_T = p[6] + p[7] * t_norm
    b2_T = p[8] + p[9] * t_norm
    
    # Numerator: physics-based components
    elastic = E_T * s
    hardening = K_T * eps_power
    yield_term = S_T * sigmoid
    num = elastic + hardening + yield_term
    
    # Denominator: rational function shape
    den = 1.0 + b1_T * s + b2_T * s ** 2
    
    # Prevent division by zero or very small numbers
    abs_den = np.abs(den)
    sign_den = np.where(den >= 0, 1.0, -1.0)
    den_safe = np.where(abs_den < 1e-8, sign_den * 1e-8, den)
    
    stress = num / den_safe
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
