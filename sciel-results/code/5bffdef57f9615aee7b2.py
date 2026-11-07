import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-based Ramberg-Osgood model for stress-strain of an aluminium rod.
    
    sigma = E(T) * strain + K(T) * strain * (strain^2 + delta)^((n(T)-1)/2) + sigma_0
    
    Where:
    E(T) = p0 + p1*T + p2*T^2       (Elastic modulus)
    K(T) = p3 + p4*T + p5*T^2       (Hardening coefficient)
    n(T) = p6 + p7*T + p8*T^2       (Hardening exponent)
    sigma_0 = p9                    (Offset)
    
    The term strain * (strain^2 + delta)^((n-1)/2) approximates |strain|^n * sign(strain)
    while remaining differentiable at strain=0.
    n is clipped to [0.1, 50.0] to ensure numerical stability.
    """
    p = params
    
    # Ensure inputs are float arrays
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    # If parameters are not finite, return zeros to avoid propagation of NaNs/Infs
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
        
    # Extract parameters
    E = p[0] + p[1] * temp + p[2] * temp**2
    K = p[3] + p[4] * temp + p[5] * temp**2
    n = p[6] + p[7] * temp + p[8] * temp**2
    sigma_0 = p[9]
    
    # Clip n to a reasonable range to prevent numerical overflow/underflow in power calculation
    n_safe = np.clip(n, 0.1, 50.0)
    
    # Small constant for numerical stability at strain=0
    delta = 1e-6
    
    s2 = strain * strain
    base = s2 + delta
    
    exponent = (n_safe - 1.0) / 2.0
    
    hardening_term = strain * np.power(base, exponent)
    
    sigma = E * strain + K * hardening_term + sigma_0
    
    return sigma
