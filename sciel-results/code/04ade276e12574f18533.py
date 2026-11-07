import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired mechanistic model for E. coli growth rate.
    
    Encodes known biology:
    - Monod saturation for substrate (s)
    - Arrhenius-type temperature dependence with optimal temp
    - Gaussian pH tolerance window
    - Logistic density-dependent inhibition (crowding)
    
    10 parameters:
    p0: max growth rate scale
    p1: Monod half-saturation constant (Ks)
    p2: optimal temperature (°C)
    p3: temperature sensitivity (beta)
    p4: pH optimum
    p5: pH width (sigma)
    p6: carrying capacity (K)
    p7: density inhibition strength
    p8: linear temp correction
    p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod substrate saturation: s / (Ks + s)
    # Ks = p1, but ensure positive via softplus-like
    Ks = np.abs(p1) + 0.1
    monod = s / (Ks + s)
    
    # Temperature: Arrhenius-like with optimum
    # Use a Gaussian around optimal temp for simplicity and differentiability
    # temp_factor = exp(-beta * (temp - T_opt)^2)
    T_opt = p2
    beta = np.abs(p3) + 0.01
    temp_factor = np.exp(-beta * (temp - T_opt) ** 2)
    
    # pH tolerance: Gaussian window
    pH_opt = p4
    pH_width = np.abs(p5) + 0.5
    pH_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_width ** 2))
    
    # Density-dependent inhibition: logistic-like
    # At low density, growth is unaffected; at high density, growth is inhibited
    K = np.abs(p6) + 1.0
    inhibition_strength = np.abs(p7) + 0.1
    # Use (1 - b/K) clipped to avoid negative, times inhibition strength
    density_factor = 1.0 - inhibition_strength * (b / K)
    # Ensure density factor stays positive (minimum 0.01)
    density_factor = np.maximum(density_factor, 0.01)
    
    # Combine all factors multiplicatively with max rate
    # Add small linear temp correction for asymmetry
    temp_linear = 1.0 + p8 * (temp - 20.0) / 20.0
    
    growth = p0 * monod * temp_factor * pH_factor * density_factor * temp_linear + p9
    
    return growth
