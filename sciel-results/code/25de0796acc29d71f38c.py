"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear composite model.
    params[0]: Amplitude of Gaussian
    params[1]: Center of Gaussian
    params[2]: Width (sigma) of Gaussian
    params[3]: Amplitude of Sine
    params[4]: Frequency of Sine
    params[5]: Phase of Sine
    params[6]: Coefficient of x^2
    params[7]: Coefficient of x
    params[8]: Coefficient of 1
    params[9]: Coefficient of x^3 (to capture cubic behavior)
    """
    
    # Ensure params are used in a way that avoids numerical instability
    # We use softplus or abs to ensure positive widths/amplitudes if needed, 
    # but since BFGS can handle negative values, we let the optimizer decide signs.
    # However, sigma must be > 0 to avoid division by zero or sign flips in exp(-x^2/2sigma^2).
    # We use a softplus-like transformation for sigma to keep it positive.
    
    a_gauss = params[0]
    center = params[1]
    # Map params[2] to a positive value > 0.1
    sigma = np.exp(params[2]) + 0.1
    
    a_sin = params[3]
    freq = params[4]
    phase = params[5]
    
    c2 = params[6]
    c1 = params[7]
    c0 = params[8]
    c3 = params[9]
    
    # Gaussian component
    gauss = a_gauss * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    # Sinusoidal component
    sine = a_sin * np.sin(freq * x + phase)
    
    # Polynomial component
    poly = c3 * x**3 + c2 * x**2 + c1 * x + c0
    
    # Combine
    y = gauss + sine + poly
    
    # Ensure finite output
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
