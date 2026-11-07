import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Model:
    mu = A * Monod(s) * Gaussian_Temp(temp) * Gaussian_pH(pH) * Logistic_Inhibition(b) + Offset
    
    Parameters:
    p[0]: Amplitude A
    p[1]: ln(K_s) for Monod
    p[2]: Temp Center
    p[3]: ln(Width_T)
    p[4]: pH Center
    p[5]: ln(Width_pH)
    p[6]: Biomass Logistic Steepness
    p[7]: Biomass Logistic Midpoint
    p[8]: Temp Skew
    p[9]: Offset
    """
    A = params[0]
    K_s = np.exp(params[1])
    
    c_T = params[2]
    w_T = np.exp(params[3])
    
    c_p = params[4]
    w_p = np.exp(params[5])
    
    k_b = params[6]
    b_0 = params[7]
    
    skew_T = params[8]
    offset = params[9]
    
    # Monod Kinetics
    monod = s / (K_s + s)
    
    # Temperature Response (Skewed Gaussian)
    temp_diff = temp - c_T
    temp_term = np.exp(-w_T * temp_diff**2 + skew_T * temp_diff)
    
    # pH Response (Gaussian)
    ph_diff = pH - c_p
    ph_term = np.exp(-w_p * ph_diff**2)
    
    # Biomass Inhibition (Logistic)
    # Use np.clip to avoid overflow warnings in exp if k_b*(b-b0) is very large
    # though 1/(1+inf) is 0, it's cleaner to handle
    z = k_b * (b - b_0)
    # Clip to prevent overflow in exp
    z = np.clip(z, -500, 500)
    biomass_term = 1.0 / (1.0 + np.exp(z))
    
    return A * monod * temp_term * ph_term * biomass_term + offset
