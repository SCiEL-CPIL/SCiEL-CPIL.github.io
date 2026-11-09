import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model:
    y = Amp * Hill(s) * Gaussian(temp) * Gaussian(pH) * (1 + C_b * b_norm) + Bias
    
    Uses a multiplicative structure with:
    - Hill kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Linear modulation for density (b)
    
    Parameters (10):
    0: log(Amp) - log of overall amplitude
    1: log(K_s) - log of substrate half-saturation constant
    2: hill_n - Hill coefficient (clipped to [0.5, 5.0])
    3: T_opt - Temperature optimum (in normalized units, center ~0.923 for 37C)
    4: log(sigma_t^2) - log of temperature variance
    5: pH_opt - pH optimum (in normalized units, center ~0.5 for pH 7)
    6: log(sigma_p^2) - log of pH variance
    7: C_b - Density coefficient
    8: Bias - additive bias
    9: unused (kept for compatibility)
    """
    
    # Normalize inputs to [0, 1]
    # b: [1, 5] -> [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    # s: [1, 40] -> [0, 1]
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    # temp: [1, 40] -> [0, 1]
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    # pH: [2, 12] -> [0, 1]
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack and transform parameters
    log_amp = np.clip(params[0], -10, 10)
    log_ks = np.clip(params[1], -10, 10)
    hill_n = np.clip(params[2], 0.5, 5.0)
    T_opt = np.clip(params[3], -0.5, 1.5)  # Allow some range around prior
    log_sig_t_sq = np.clip(params[4], -10, 10)
    pH_opt = np.clip(params[5], -0.5, 1.5)  # Allow some range around prior
    log_sig_p_sq = np.clip(params[6], -10, 10)
    C_b = params[7]
    bias = params[8]
    # params[9] is unused
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    sigma_t_sq = np.exp(log_sig_t_sq)
    sigma_p_sq = np.exp(log_sig_p_sq)
    
    # Calculate terms
    
    # Substrate: Hill Function
    # H(s) = s_norm^n / (K_s^n + s_norm^n)
    # Use s_norm for consistency
    s_term = (s_norm ** hill_n) / (K_s ** hill_n + s_norm ** hill_n + 1e-9)
    
    # Temperature: Gaussian
    # Center prior: 0.923 (for 37C in [1,40] range)
    t_center = 0.923 + T_opt
    temp_term = np.exp(-((temp_norm - t_center) ** 2) / (2.0 * sigma_t_sq + 1e-9))
    
    # pH: Gaussian
    # Center prior: 0.5 (for pH 7 in [2,12] range)
    ph_center = 0.5 + pH_opt
    ph_term = np.exp(-((ph_norm - ph_center) ** 2) / (2.0 * sigma_p_sq + 1e-9))
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine multiplicatively
    result = Amp * s_term * temp_term * ph_term * dens_term + bias
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
