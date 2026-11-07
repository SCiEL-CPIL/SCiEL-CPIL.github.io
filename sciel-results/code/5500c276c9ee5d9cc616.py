import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod combining:
    - Ramberg-Osgood style hardening (fractional power of strain)
    - Smooth yield transition (sigmoid)
    - Temperature-dependent elastic modulus and hardening
    - Linear in parameters for robust BFGS convergence
    
    Structure:
    sigma = E(T) * eps + H(T) * eps^n + S(T) * sigmoid(eps) + offset
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2  (elastic modulus, temperature-dependent)
    - H(T) = p3 + p4*T + p5*T^2  (hardening coefficient)
    - S(T) = p6 + p7*T           (yield transition amplitude)
    - p8: offset
    - p9: additional cubic-like term for large strain behavior
    """
    s = strain
    t = temp
    p = params
    
    # Normalize temperature to [0, 1]
    t_norm = (t - 0.06667) / 0.93333
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Use absolute strain for fractional power (handle negative strains)
    eps_abs = np.abs(s)
    
    # Hardening exponent for aluminium (~0.1-0.2)
    n = 0.15
    
    # Fractional power term for plastic hardening
    # Add small epsilon to avoid 0^0 issues
    eps_safe = np.maximum(eps_abs, 1e-12)
    eps_power = np.power(eps_safe, n)
    
    # Sigmoid for smooth yield transition
    # Yield strain for aluminium is typically ~0.001-0.005
    yield_center = 0.003
    yield_width = 0.008
    # Use tanh instead of sigmoid for numerical stability
    sigmoid = 0.5 * (1.0 + np.tanh((s - yield_center) / yield_width))
    
    # Temperature-dependent coefficients
    E_T = p[0] + p[1] * t_norm + p[2] * t_norm ** 2
    H_T = p[3] + p[4] * t_norm + p[5] * t_norm ** 2
    S_T = p[6] + p[7] * t_norm
    
    # Elastic component
    elastic = E_T * s
    
    # Hardening component (Ramberg-Osgood style)
    hardening = H_T * eps_power
    
    # Yield transition component
    yield_term = S_T * sigmoid
    
    # Additional higher-order term for large strains
    extra = p[9] * s * (s ** 2) * (1.0 - t_norm * 0.5)
    
    # Offset
    offset = p[8] * np.ones_like(s)
    
    stress = elastic + hardening + yield_term + extra + offset
    
    return stress
