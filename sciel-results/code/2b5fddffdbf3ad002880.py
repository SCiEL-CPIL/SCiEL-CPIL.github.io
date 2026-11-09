import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using sum-of-sines (Fourier) basis for temperature and pH,
    with Hill kinetics for substrate and linear modulation for density.
    
    y = Amp * Hill(s) * TempFourier * PhFourier * (1 + C_b * b_norm) + Bias
    
    Parameters (10):
    0: log(Amp) - log of overall amplitude
    1: log(K_s) - log of substrate half-saturation constant
    2: hill_n - Hill coefficient (clipped to [0.5, 5.0])
    3: temp_freq - frequency for temperature sine term
    4: temp_phase - phase for temperature sine term
    5: ph_freq - frequency for pH sine term
    6: ph_phase - phase for pH sine term
    7: C_b - Density coefficient
    8: Bias - additive bias
    9: temp_amp - amplitude modulation for temperature term
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
    temp_freq = np.clip(params[3], 0.1, 20.0)
    temp_phase = np.clip(params[4], -np.pi, np.pi)
    ph_freq = np.clip(params[5], 0.1, 20.0)
    ph_phase = np.clip(params[6], -np.pi, np.pi)
    C_b = params[7]
    bias = params[8]
    temp_amp = np.clip(params[9], -5.0, 5.0)
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    
    # Substrate: Hill Function
    s_term = (s_norm ** hill_n) / (K_s ** hill_n + s_norm ** hill_n + 1e-9)
    
    # Temperature: Sum-of-sines basis
    # Use a combination of sine and cosine to capture the bell-shaped response
    temp_term = np.sin(temp_freq * temp_norm * np.pi + temp_phase) + \
                0.5 * np.cos(2 * temp_freq * temp_norm * np.pi + 2 * temp_phase)
    temp_term = 1.0 + temp_amp * temp_term
    
    # pH: Sum-of-sines basis
    ph_term = np.sin(ph_freq * ph_norm * np.pi + ph_phase) + \
              0.5 * np.cos(2 * ph_freq * ph_norm * np.pi + 2 * ph_phase)
    ph_term = 1.0 + temp_amp * ph_term
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine multiplicatively
    result = Amp * s_term * temp_term * ph_term * dens_term + bias
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
