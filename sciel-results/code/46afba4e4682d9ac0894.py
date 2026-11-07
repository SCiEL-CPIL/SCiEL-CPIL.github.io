import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Ensure positive parameters where necessary
    eps = 1e-8
    
    # p[0]: Max growth rate (mu_max)
    # Can be negative, but likely positive for growth.
    
    # p[1]: Substrate saturation constant (Ks)
    # Monod term: s / (s + Ks)
    # If Ks is negative, denominator can be zero. Use softplus or abs?
    # Let's use a smooth positive mapping for Ks to avoid discontinuities.
    # Ks_eff = exp(p[1]) ensures Ks > 0.
    # However, BFGS might struggle if the scale is very different.
    # Let's try: Ks = p[1]**2 + eps. This is smooth and non-negative.
    # Or better, use sigmoid to map to a reasonable range?
    # Let's stick with a robust positive mapping.
    # Ks = 1.0 / (1.0 + np.exp(-p[1])) maps to (0, 1).
    # But Ks might need to be larger than 1.
    # Let's use: Ks = np.exp(p[1])
    
    # p[2]: Temperature optimum
    # p[3]: Temperature width (inverse variance)
    # Gaussian: exp(-width * (t - opt)^2)
    # width should be positive. width = np.exp(p[3])
    
    # p[4]: pH optimum
    # p[5]: pH width
    # width = np.exp(p[5])
    
    # p[6]: Biomass inhibition strength
    # Term: exp(-strength * b_norm) or exp(-strength * b_norm^2)
    # strength = np.exp(p[6])
    
    # p[7]: Interaction term coefficient (s * t)
    # p[8]: Interaction term coefficient (s * ph)
    # p[9]: Bias
    
    # Let's refine the structure to be more flexible.
    # Growth rate = mu_max * f(s) * f(t) * f(ph) * f(b) + bias
    # f(s) = s / (s + Ks)
    # f(t) = exp(-w_t * (t - t_opt)^2)
    # f(ph) = exp(-w_ph * (ph - ph_opt)^2)
    # f(b) = exp(-w_b * b)
    
    # Parameters:
    # p[0]: mu_max
    # p[1]: Ks (log scale)
    # p[2]: t_opt
    # p[3]: w_t (log scale)
    # p[4]: ph_opt
    # p[5]: w_ph (log scale)
    # p[6]: w_b (log scale)
    # p[7]: bias
    # p[8]: linear temp correction
    # p[9]: linear ph correction
    
    # Calculate terms
    Ks = np.exp(p[1])
    term_s = s_norm / (s_norm + Ks)
    
    t_opt = p[2]
    w_t = np.exp(p[3])
    term_t = np.exp(-w_t * (t_norm - t_opt)**2)
    
    ph_opt = p[4]
    w_ph = np.exp(p[5])
    term_ph = np.exp(-w_ph * (ph_norm - ph_opt)**2)
    
    w_b = np.exp(p[6])
    term_b = np.exp(-w_b * b_norm)
    
    main_term = p[0] * term_s * term_t * term_ph * term_b
    
    # Add linear corrections to capture non-multiplicative effects
    # These help if the optimal temperature/pH shift with other variables
    linear_corr = p[7] + p[8] * t_norm + p[9] * ph_norm
    
    y = main_term + linear_corr
    
    return y
