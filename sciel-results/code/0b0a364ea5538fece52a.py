import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model:
    y = Amp * Hill(s) * TempResponse * PhResponse * (1 + C_b * b_norm) + Bias
    
    Uses a multiplicative structure with:
    - Hill kinetics for substrate (s)
    - Skewed Gaussian response for temperature (temp)
    - Skewed Gaussian response for pH
    - Linear modulation for density (b)
    
    Parameters (10):
    0: log(Amp) - log of overall amplitude
    1: log(K_s) - log of substrate half-saturation constant
    2: hill_n - Hill coefficient (clipped to [0.5, 5.0])
    3: T_opt - Temperature optimum offset (added to prior 0.923)
    4: log(sigma_t) - log of temperature standard deviation
    5: pH_opt - pH optimum offset (added to prior 0.5)
    6: log(sigma_p) - log of pH standard deviation
    7: C_b - Density coefficient
    8: Bias - additive bias
    9: temp_skew - skew coefficient for temperature Gaussian
    10: ph_skew - skew coefficient for pH Gaussian
    """
    
    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack and transform parameters
    log_amp = np.clip(params[0], -10, 10)
    log_ks = np.clip(params[1], -10, 10)
    hill_n = np.clip(params[2], 0.5, 5.0)
    T_opt_offset = np.clip(params[3], -0.5, 1.5)
    log_sig_t = np.clip(params[4], -10, 10)
    pH_opt_offset = np.clip(params[5], -0.5, 1.5)
    log_sig_p = np.clip(params[6], -10, 10)
    C_b = params[7]
    bias = params[8]
    temp_skew = np.clip(params[9], -5.0, 5.0)
    ph_skew = np.clip(params[10] if len(params) > 10 else 0.0, -5.0, 5.0)
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    sigma_t = np.exp(log_sig_t)
    sigma_p = np.exp(log_sig_p)
    
    # Substrate: Hill Function
    s_term = (s_norm ** hill_n) / (K_s ** hill_n + s_norm ** hill_n + 1e-9)
    
    # Temperature: Skewed Gaussian
    t_center = 0.923 + T_opt_offset
    t_diff = temp_norm - t_center
    temp_term = np.exp(-0.5 * (t_diff / sigma_t) ** 2 + temp_skew * t_diff)
    
    # pH: Skewed Gaussian
    ph_center = 0.5 + pH_opt_offset
    ph_diff = ph_norm - ph_center
    ph_term = np.exp(-0.5 * (ph_diff / sigma_p) ** 2 + ph_skew * ph_diff)
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine multiplicatively
    result = Amp * s_term * temp_term * ph_term * dens_term + bias
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
