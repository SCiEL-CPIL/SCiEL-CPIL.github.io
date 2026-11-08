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
    Ki = params[2] # Inhibition constant for Haldane
    Topt = params[3]
    sigma_T = params[4] # Width for temp
    pHopt = params[5]
    sigma_P = params[6] # Width for pH
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
    # This handles saturation AND inhibition
    monod_s = s / (Ks_safe + s + (s * s) / Ki_safe)
    
    # Gaussian for Temperature: exp(-0.5 * ((T - Topt) / sigma_T)^2)
    # Using stable exp
    dT = temp - Topt
    temp_factor = np.exp(-0.5 * (dT / sigma_T_safe) ** 2)
    
    # Gaussian for pH: exp(-0.5 * ((pH - pHopt) / sigma_P)^2)
    dP = pH - pHopt
    ph_factor = np.exp(-0.5 * (dP / sigma_P_safe) ** 2)
    
    # Density inhibition: 1 / (1 + (b/Kd)^2)
    density_factor = 1.0 / (1.0 + (b / Kd_safe) ** 2)
    
    # Base multiplicative term
    base_growth = mu_max * monod_s * temp_factor * ph_factor * density_factor
    
    # Structural Change: Instead of simple additive offsets, use coupled terms
    # This allows the substrate level to modify the temperature sensitivity or similar
    # w1 * monod_s * temp_factor * ph_factor
    # w2 * density_factor * temp_factor
    
    # Let's try a "flexible" additive component that is also gated by the main factors
    # to keep the zero-growth state at low s or bad T/pH
    interaction_term = w1 * monod_s * temp_factor * ph_factor * density_factor
    interaction_term += w2 * (temp_factor * ph_factor) * density_factor
    
    growth = base_growth + interaction_term
    
    # Return
    return growth
