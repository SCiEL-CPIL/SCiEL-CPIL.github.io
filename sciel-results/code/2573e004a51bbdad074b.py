import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Voce-type exponential hardening
    formulation combined with rational function structure for flexibility.
    
    sigma = E(T)*eps + [sigma_y(T) + Q(T)*(1 - exp(-eps/H(T)))] * (1 + p8*eps*T) / (1 + p9*eps^2)
    
    Where:
    E(T) = p0 + p1*T + p2*T^2   (elastic modulus, temperature-dependent)
    sigma_y(T) = p3 + p4*T      (yield stress, temperature-dependent)
    Q(T) = p5 + p6*T            (saturation stress, temperature-dependent)
    H(T) = p7 + p8*T            (hardening modulus, temperature-dependent)
    p9: rational correction term for higher-order behavior
    
    The Voce form naturally captures asymptotic saturation of stress at high strains,
    while the rational denominator provides additional shape flexibility.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    # Elastic modulus
    E = p0 + p1 * T + p2 * T**2
    
    # Yield stress (linear in T)
    sigma_y = p3 + p4 * T
    
    # Saturation stress (linear in T)
    Q = p5 + p6 * T
    
    # Hardening modulus (linear in T), ensure positive for stable exponential
    H = p7 + p8 * T
    H_safe = np.where(np.abs(H) > 1e-8, H, 1e-8)
    
    # Voce-type plastic term: Q(T) * (1 - exp(-eps / H(T)))
    # For negative strains, we want the plastic term to also be negative (odd symmetry)
    # Use sign(eps) * (1 - exp(-|eps|/H)) to maintain odd symmetry
    abs_u = np.abs(u)
    sign_u = np.sign(u)
    
    # Compute exponential term safely
    # For large |eps|/H, exp(-|eps|/H) -> 0, so (1 - exp) -> 1
    # For small |eps|/H, exp(-|eps|/H) ~ 1 - |eps|/H, so (1 - exp) ~ |eps|/H
    ratio = abs_u / H_safe
    # Clip ratio to avoid overflow in exp for very large values
    ratio_clipped = np.clip(ratio, 0.0, 50.0)
    exp_term = np.exp(-ratio_clipped)
    voce_term = Q * (1.0 - exp_term)
    
    # Apply sign for odd symmetry
    plastic = sign_u * voce_term
    
    # Elastic + plastic
    sigma = E * u + plastic
    
    # Rational correction for additional flexibility
    # (1 + p8*u*T) / (1 + p9*u^2)
    num_corr = 1.0 + p8 * u * T
    den_corr = 1.0 + p9 * u**2
    # Safe division
    den_safe = np.where(np.abs(den_corr) > 1e-12, den_corr, 1e-12)
    sigma = sigma * num_corr / den_safe
    
    return sigma
