import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Fourier basis for temp/pH and Hill for substrate.
    
    Structure:
    y = Amp * Hill(s) * TempFourier * PhFourier * (1 + C_b * b_norm) + Bias
    
    TempFourier = 1 + a1*cos(w1*t) + a2*sin(w1*t) + a3*cos(w2*t) + a4*sin(w2*t)
    PhFourier = 1 + a5*cos(w3*p) + a6*sin(w3*p)
    
    Parameters (10):
    0: log(Amp)
    1: log(K_s)
    2: hill_n (clipped [0.5, 5])
    3: T_opt (temperature center, normalized)
    4: log(sigma_t)
    5: pH_opt (pH center, normalized)
    6: log(sigma_p)
    7: C_b (density coefficient)
    8: Bias
    9: interaction_strength
    """
    
    # Deadline guard
    start_time = time.time()
    
    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack parameters
    p0 = np.clip(params[0], -10, 10)
    p1 = np.clip(params[1], -10, 10)
    p2 = np.clip(params[2], 0.5, 5.0)
    p3 = np.clip(params[3], -1.0, 2.0)  # T_opt
    p4 = np.clip(params[4], -10, 10)    # log(sigma_t)
    p5 = np.clip(params[5], -1.0, 2.0)  # pH_opt
    p6 = np.clip(params[6], -10, 10)    # log(sigma_p)
    p7 = params[7]                       # C_b
    p8 = params[8]                       # Bias
    p9 = np.clip(params[9], -2.0, 2.0)  # interaction strength
    
    log_amp = p0
    log_ks = p1
    hill_n = p2
    T_opt = p3
    log_sig_t = p4
    pH_opt = p5
    log_sig_p = p6
    C_b = p7
    bias = p8
    inter = p9
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    sigma_t = np.exp(log_sig_t)
    sigma_p = np.exp(log_sig_p)
    
    # Substrate: Hill Function
    s_n = s_norm ** hill_n
    K_n = K_s ** hill_n
    s_term = s_n / (K_n + s_n + 1e-9)
    
    # Temperature: Skewed Gaussian with Fourier modulation
    t_diff = temp_norm - T_opt
    t_exponent = -0.5 * (t_diff / (sigma_t + 1e-6)) ** 2 + inter * t_diff
    t_exponent = np.clip(t_exponent, -100, 100)
    temp_base = np.exp(t_exponent)
    
    # Add Fourier harmonics to capture non-Gaussian features
    w1 = 2.0 * np.pi
    w2 = 4.0 * np.pi
    temp_fourier = 1.0 + 0.1 * np.cos(w1 * t_diff) + 0.05 * np.sin(w1 * t_diff)
    temp_term = temp_base * temp_fourier
    
    # pH: Skewed Gaussian with Fourier modulation
    ph_diff = ph_norm - pH_opt
    ph_exponent = -0.5 * (ph_diff / (sigma_p + 1e-6)) ** 2 + inter * ph_diff
    ph_exponent = np.clip(ph_exponent, -100, 100)
    ph_base = np.exp(ph_exponent)
    
    w3 = 2.0 * np.pi
    ph_fourier = 1.0 + 0.1 * np.cos(w3 * ph_diff) + 0.05 * np.sin(w3 * ph_diff)
    ph_term = ph_base * ph_fourier
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine multiplicatively
    result = Amp * s_term * temp_term * ph_term * dens_term + bias
    
    # Check deadline
    if time.time() - start_time > 140:
        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
