import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier-Basis Periodic Model for E. coli growth rate.
    
    Uses sums of sine and cosine terms with fixed frequencies for temperature and pH
    to capture asymmetric, multi-modal, or non-Gaussian environmental response curves.
    Substrate (s) uses a Monod-like term, and density (b) uses a logistic inhibition.
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Temp frequency 1 (fixed at 1, but we use params for amplitudes)
    p3: Temp frequency 2 (fixed at 2, but we use params for amplitudes)
    p4: pH frequency 1 (fixed at 1, but we use params for amplitudes)
    p5: pH frequency 2 (fixed at 2, but we use params for amplitudes)
    p6: Temp sine amplitude 1
    p7: Temp cosine amplitude 1
    p8: pH sine amplitude 1
    p9: pH cosine amplitude 1
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Substrate term: Monod-like
    Ks = np.exp(p1)
    substrate = s / (Ks + s)
    
    # Density inhibition: logistic
    # b is in [1, 4.998], so we normalize it
    b_norm = b / 5.0
    density_inhibition = 1.0 / (1.0 + np.exp(3.0 * (b_norm - 0.5)))
    
    # Temperature Fourier terms
    # Fixed frequencies: 1 and 2
    temp_sin1 = np.sin(2 * np.pi * temp / 10.0)
    temp_cos1 = np.cos(2 * np.pi * temp / 10.0)
    temp_sin2 = np.sin(2 * np.pi * 2 * temp / 10.0)
    temp_cos2 = np.cos(2 * np.pi * 2 * temp / 10.0)
    
    # pH Fourier terms
    pH_sin1 = np.sin(2 * np.pi * pH / 2.0)
    pH_cos1 = np.cos(2 * np.pi * pH / 2.0)
    pH_sin2 = np.sin(2 * np.pi * 2 * pH / 2.0)
    pH_cos2 = np.cos(2 * np.pi * 2 * pH / 2.0)
    
    # Combine temperature terms
    temp_response = p6 * temp_sin1 + p7 * temp_cos1 + 0.5 * p6 * temp_sin2 + 0.5 * p7 * temp_cos2
    
    # Combine pH terms
    pH_response = p8 * pH_sin1 + p9 * pH_cos1 + 0.5 * p8 * pH_sin2 + 0.5 * p9 * pH_cos2
    
    # Final prediction
    growth = p0 * substrate * density_inhibition * (1.0 + temp_response) * (1.0 + pH_response)
    
    return growth
