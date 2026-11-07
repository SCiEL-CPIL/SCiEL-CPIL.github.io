"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    Inputs are scaled to [-1, 1] to improve numerical conditioning.
    """
    # Scaling parameters based on training ranges
    # x in [-0.5728, 0.5275] -> center -0.02265, half-range 0.55015
    # v in [-0.1754, 0.2821] -> center 0.05335, half-range 0.22875
    
    cx = -0.02265
    hx = 0.55015
    cv = 0.05335
    hv = 0.22875
    
    # Scale to [-1, 1]
    u = (x - cx) / hx
    w = (v - cv) / hv
    
    # Clip to [-1, 1] to handle slight out-of-range values and ensure stability
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Chebyshev polynomials
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2 * u * u - 1
    T3_u = 4 * u * u * u - 3 * u
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2 * w * w - 1
    T3_w = 4 * w * w * w - 3 * w
    
    # Construct the 10 basis terms
    # Order:
    # 0: T0(u)*T0(w)
    # 1: T1(u)*T0(w)
    # 2: T0(u)*T1(w)
    # 3: T2(u)*T0(w)
    # 4: T0(u)*T2(w)
    # 5: T1(u)*T1(w)
    # 6: T3(u)*T0(w)
    # 7: T0(u)*T3(w)
    # 8: T2(u)*T1(w)
    # 9: T1(u)*T2(w)
    
    term0 = T0_u * T0_w
    term1 = T1_u * T0_w
    term2 = T0_u * T1_w
    term3 = T2_u * T0_w
    term4 = T0_u * T2_w
    term5 = T1_u * T1_w
    term6 = T3_u * T0_w
    term7 = T0_u * T3_w
    term8 = T2_u * T1_w
    term9 = T1_u * T2_w
    
    # Combine with parameters
    return (params[0] * term0 + 
            params[1] * term1 + 
            params[2] * term2 + 
            params[3] * term3 + 
            params[4] * term4 + 
            params[5] * term5 + 
            params[6] * term6 + 
            params[7] * term7 + 
            params[8] * term8 + 
            params[9] * term9)
# EVOLVE-BLOCK-END
