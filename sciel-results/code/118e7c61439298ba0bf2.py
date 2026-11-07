"""Symbolic regression for E. coli growth rate - Biologically-inspired Gaussian-Monod model."""
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: mu = mu_max * Monod(s) * Gaussian(temp) * Gaussian(pH) * (1 + c_b * b_norm) + bias
    
    Parameters (10 total):
    p[0]: mu_max
    p[1]: log_Ks
    p[2]: temp_center
    p[3]: log_temp_sigma
    p[4]: temp_amp
    p[5]: ph_center
    p[6]: log_ph_sigma
    p[7]: ph_amp
    p[8]: biomass_coeff
    p[9]: bias
    """
    p = params
    
    mu_max = p[0]
    log_Ks = p[1]
    temp_c = p[2]
    log_temp_sig = p[3]
    temp_amp = p[4]
    
    ph_c = p[5]
    log_ph_sig = p[6]
    ph_amp = p[7]
    
    c_b = p[8]
    bias = p[9]
    
    # Normalization based on prompt ranges
    # b: [1, 5]
    b_norm = (b - 1.0) / 4.0
    b_norm = np.clip(b_norm, 0.0, 1.0)
    
    # s: [1, 5] (Assuming prompt order b, s, temp, pH)
    # If s is actually [1, 40], this normalization is wrong.
    # To be safe, I will use a wider range for s normalization if possible, 
    # but Monod is scale-invariant to some degree if Ks is fitted.
    # However, if I normalize s to [0,1] using [1,5], and the data is [1,40], 
    # s_norm will be > 1 for most data.
    # Monod: s_norm / (Ks + s_norm). If s_norm is large, it saturates.
    # This is fine.
    # But if I normalize by 39 (seed), and data is [1,5], s_norm is [0, 0.1].
    # Let's stick to the prompt's explicit ranges.
    s_norm = (s - 1.0) / 4.0 
    s_norm = np.clip(s_norm, 0.0, 10.0) # Allow some overshoot if range is slightly off
    
    # temp: [1, 40]
    temp_norm = (temp - 1.0) / 39.0
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    
    # pH: [2, 12]
    ph_norm = (pH - 2.0) / 10.0
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Monod Term
    Ks = np.exp(log_Ks) + 1e-8
    monod_s = s_norm / (Ks + s_norm + 1e-12)
    
    # Gaussian Terms
    # Using raw values for centers to allow fitting in physical space
    # temp is in [1, 40], pH is in [2, 12]
    
    # Ensure sigma is positive and not too small/large
    temp_sigma = np.exp(log_temp_sig) + 1e-4
    ph_sigma = np.exp(log_ph_sig) + 1e-4
    
    # Gaussian for Temperature
    # temp is original value
    temp_diff = (temp - temp_c) / temp_sigma
    temp_factor = temp_amp * np.exp(-0.5 * temp_diff**2)
    
    # Gaussian for pH
    ph_diff = (pH - ph_c) / ph_sigma
    ph_factor = ph_amp * np.exp(-0.5 * ph_diff**2)
    
    # Biomass Term
    b_term = 1.0 + c_b * b_norm
    
    # Combine
    mu = mu_max * monod_s * temp_factor * ph_factor * b_term + bias
    
    # Safety clip
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu
