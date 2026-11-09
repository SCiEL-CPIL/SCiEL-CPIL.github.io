"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum-of-sines (Fourier) basis model for E. coli growth rate.
    
    Uses learnable frequencies and phases for each input to capture
    periodic/oscillatory constraints (pH optima, temperature optima).
    
    Parameters (length 10):
    p[0]: Bias
    p[1]: Frequency scale for temp
    p[2]: Frequency scale for pH
    p[3]: Frequency scale for substrate (log-scaled)
    p[4]: Phase offset for temp
    p[5]: Phase offset for pH
    p[6]: Coefficient for sin(temp)
    p[7]: Coefficient for sin(pH)
    p[8]: Coefficient for sin(substrate) * cos(temp)
    p[9]: Overall amplitude/gain
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to approximately [0, 1]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 48.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    # Safely compute frequencies (avoid zero or negative)
    f_t = np.maximum(0.1, p1)
    f_ph = np.maximum(0.1, p2)
    f_s = np.maximum(0.1, p3)
    
    # Fourier components
    sin_t = np.sin(f_t * t_n * np.pi + p4)
    sin_ph = np.sin(f_ph * ph_n * np.pi + p5)
    sin_s = np.sin(f_s * np.log1p(s_n * 10.0) + 0.5)
    cos_t = np.cos(f_t * t_n * np.pi + p4)
    
    # Combine with interactions
    y = p0
    y += p6 * sin_t
    y += p7 * sin_ph
    y += p8 * (sin_s * cos_t)
    y += p9 * (sin_t * sin_ph)
    
    # Add a linear trend for monotonic parts
    y += 0.5 * p9 * t_n
    y += 0.3 * p9 * ph_n
    
    return y
# EVOLVE-BLOCK-END
