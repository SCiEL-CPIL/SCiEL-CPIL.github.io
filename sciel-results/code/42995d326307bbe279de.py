import numpy as np
import time

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
    10: ph_skew - skew coefficient for pH Gaussian (if available, else 0)
    """
    
    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack and transform parameters
    # params is expected to be length 10, but we handle length 11 gracefully
    p0 = np.clip(params[0], -10, 10)
    p1 = np.clip(params[1], -10, 10)
    p2 = np.clip(params[2], 0.5, 5.0)
    p3 = np.clip(params[3], -0.5, 1.5)
    p4 = np.clip(params[4], -10, 10)
    p5 = np.clip(params[5], -0.5, 1.5)
    p6 = np.clip(params[6], -10, 10)
    p7 = params[7]
    p8 = params[8]
    p9 = np.clip(params[9], -5.0, 5.0)
    
    # Handle potential 11th parameter for separate pH skew
    if len(params) > 10:
        p10 = np.clip(params[10], -5.0, 5.0)
    else:
        p10 = 0.0
    
    log_amp = p0
    log_ks = p1
    hill_n = p2
    T_opt_offset = p3
    log_sig_t = p4
    pH_opt_offset = p5
    log_sig_p = p6
    C_b = p7
    bias = p8
    temp_skew = p9
    ph_skew = p10
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    sigma_t = np.exp(log_sig_t)
    sigma_p = np.exp(log_sig_p)
    
    # Substrate: Hill Function
    # s_term = s_norm^n / (K_s^n + s_norm^n)
    # Using log-sum-exp style stability or just direct with clipping
    s_n = s_norm ** hill_n
    K_n = K_s ** hill_n
    s_term = s_n / (K_n + s_n + 1e-9)
    
    # Temperature: Skewed Gaussian
    t_center = 0.923 + T_opt_offset
    t_diff = temp_norm - t_center
    # Clip exponent to avoid overflow/underflow issues before exp
    t_exponent = -0.5 * (t_diff / sigma_t) ** 2 + temp_skew * t_diff
    t_exponent = np.clip(t_exponent, -100, 100)
    temp_term = np.exp(t_exponent)
    
    # pH: Skewed Gaussian
    ph_center = 0.5 + pH_opt_offset
    ph_diff = ph_norm - ph_center
    ph_exponent = -0.5 * (ph_diff / sigma_p) ** 2 + ph_skew * ph_diff
    ph_exponent = np.clip(ph_exponent, -100, 100)
    ph_term = np.exp(ph_exponent)
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine multiplicatively
    result = Amp * s_term * temp_term * ph_term * dens_term + bias
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
