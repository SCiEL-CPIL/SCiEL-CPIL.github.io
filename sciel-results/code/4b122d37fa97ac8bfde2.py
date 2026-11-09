import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # Parameters
    mu_max = params[0]
    Ks = params[1]
    Ki = params[2]  # Haldane inhibition constant
    Topt = params[3]
    sigma_T = params[4]  # Width for temp Gaussian
    pHopt = params[5]
    sigma_P = params[6]  # Width for pH Gaussian
    Kd = params[7]
    w1 = params[8]
    w2 = params[9]
    
    # Safe parameters
    Ks_safe = np.clip(Ks, 1e-6, 1e6)
    Ki_safe = np.clip(Ki, 1e-6, 1e6)
    sigma_T_safe = np.clip(sigma_T, 1e-6, 1e6)
    sigma_P_safe = np.clip(sigma_P, 1e-6, 1e6)
    Kd_safe = np.clip(Kd, 1e-6, 1e6)
    
    # Haldane for substrate: s / (Ks + s + s^2/Ki)
    haldane_s = s / (Ks_safe + s + (s ** 2) / Ki_safe)
    
    # Gaussian bell for Temperature: exp(-sigma_T * (T - Topt)^2)
    dT = temp - Topt
    temp_factor = np.exp(-sigma_T_safe * dT * dT)
    
    # Gaussian bell for pH: exp(-sigma_P * (pH - pHopt)^2)
    dP = pH - pHopt
    ph_factor = np.exp(-sigma_P_safe * dP * dP)
    
    # Density inhibition: 1 / (1 + (b/Kd)^2)
    density_factor = 1.0 / (1.0 + (b / Kd_safe) ** 2)
    
    # Base term
    growth = mu_max * haldane_s * temp_factor * ph_factor * density_factor
    
    # Interactions
    growth += w1 * haldane_s * temp_factor
    growth += w2 * ph_factor * density_factor
    
    return growth
