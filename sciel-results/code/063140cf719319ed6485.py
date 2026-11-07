import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Voce-type exponential hardening model for stress-strain of an aluminium rod.
    
    sigma = E(T)*strain + (sigma_y(T) + Q(T)*(1 - exp(-|strain|/H(T)))) * smooth_sign(strain)
    
    Where:
    sigma_y(T) = p0 + p1*T          (Yield stress, linear in temperature)
    Q(T)       = p2 + p3*T          (Saturation stress, linear in temperature)
    H(T)       = p4 + p5*T          (Hardening modulus, linear in temperature)
    E(T)       = p6 + p7*T          (Elastic modulus, linear in temperature)
    p8, p9:    Additional correction terms for flexibility
    
    smooth_sign uses tanh for differentiability, enabling BFGS to converge well.
    The tanh width is tied to H so the transition scales with the hardening length.
    """
    p = params
    
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    abs_strain = np.abs(strain)
    
    # Temperature-dependent material parameters
    sigma_y = p[0] + p[1] * T
    Q = p[2] + p[3] * T
    H = p[4] + p[5] * T
    E = p[6] + p[7] * T
    
    # Clip H to a safe range to avoid numerical issues
    H_safe = np.clip(H, 1e-3, 1e4)
    
    # Exponential hardening term with clipped exponent
    exponent = -abs_strain / H_safe
    exponent = np.clip(exponent, -40.0, 0.0)
    exp_term = np.exp(exponent)
    voce = Q * (1.0 - exp_term)
    
    # Smooth sign via tanh: width scales with H so the elastic-plastic
    # transition is smooth and differentiable everywhere.
    # tanh(strain / w) -> 0 at strain=0, +/-1 for |strain| >> w
    w = np.maximum(H_safe * 0.05, 1e-4)
    smooth_sign = np.tanh(strain / w)
    
    # Elastic part
    elastic = E * strain
    
    # Plastic part: yield + Voce hardening, gated by smooth sign
    plastic = (sigma_y + voce) * smooth_sign
    
    # Correction terms for additional flexibility
    correction = p[8] * strain + p[9] * strain * T
    
    sigma = elastic + plastic + correction
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
