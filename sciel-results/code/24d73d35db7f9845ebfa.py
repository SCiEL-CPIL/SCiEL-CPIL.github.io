"""E. coli growth rate model with Fourier-basis periodic environmental response."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using Fourier basis for temperature and pH to capture
    asymmetric, multi-modal environmental responses. The basis is fixed and the
    10 parameters are linear coefficients, making this a convex least-squares
    problem solvable by BFGS.
    
    Basis functions:
      - Monod substrate saturation
      - Fourier temperature terms (sin/cos at fixed frequencies)
      - Fourier pH terms (sin/cos at fixed frequencies)
      - Key interaction terms
      - Constant
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological constants
    Ks = 5.0        # Substrate half-saturation
    T_center = 37.0 # Temperature center
    T_scale = 10.0  # Temperature scale for normalization
    pH_center = 7.0 # pH center
    pH_scale = 2.0  # pH scale for normalization
    
    # Core basis functions
    monod_s = s / (Ks + s)
    
    # Normalized inputs for Fourier terms
    t_norm = (temp - T_center) / T_scale
    ph_norm = (pH - pH_center) / pH_scale
    
    # Fourier basis for temperature (frequencies: 1, 2, 3)
    sin_t1 = np.sin(2.0 * np.pi * t_norm)
    cos_t1 = np.cos(2.0 * np.pi * t_norm)
    sin_t2 = np.sin(4.0 * np.pi * t_norm)
    cos_t2 = np.cos(4.0 * np.pi * t_norm)
    
    # Fourier basis for pH (frequencies: 1, 2)
    sin_ph1 = np.sin(2.0 * np.pi * ph_norm)
    cos_ph1 = np.cos(2.0 * np.pi * ph_norm)
    sin_ph2 = np.sin(4.0 * np.pi * ph_norm)
    cos_ph2 = np.cos(4.0 * np.pi * ph_norm)
    
    # Approximate Gaussian-like envelope from Fourier terms for stability
    # Use a simple monotonic decay based on distance from center
    t_env = np.exp(-1.5 * np.abs(t_norm))
    ph_env = np.exp(-2.0 * np.abs(ph_norm))
    
    # Combined environmental response using Fourier + envelope
    env_temp = t_env * (cos_t1 + 0.5 * cos_t2)
    env_ph = ph_env * (cos_ph1 + 0.5 * cos_ph2)
    
    # Interaction terms
    monod_env_t = monod_s * env_temp
    monod_env_p = monod_s * env_ph
    env_tp = env_temp * env_ph
    monod_env_tp = monod_s * env_tp
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * monod_env_tp +
        p1 * monod_env_t +
        p2 * monod_env_p +
        p3 * env_tp +
        p4 * monod_s +
        p5 * env_temp +
        p6 * env_ph +
        p7 * monod_s * t_env * ph_env +
        p8 * t_env * ph_env +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
