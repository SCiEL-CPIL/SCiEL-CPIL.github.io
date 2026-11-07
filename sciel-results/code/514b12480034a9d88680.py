"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    Strategy: Physically-motivated Duffing-van der Pol style basis.
    Prioritizes odd powers (x, x^3, v, v^3) and cross-couplings (xv, x^2v) 
    along with time-dependent forcing terms, assuming the oscillator dynamics 
    are dominated by restoring, damping, and driving forces rather than 
    generic polynomial noise.
    
    Basis:
    p[0] * x
    p[1] * x^3
    p[2] * v
    p[3] * v^3
    p[4] * x * v
    p[5] * x^2 * v
    p[6] * np.cos(t)
    p[7] * np.sin(t)
    p[8] * x * np.cos(t)
    p[9] * v * np.sin(t)
    """
    p = params
    
    # Extract parameters
    c_x = p[0]
    c_x3 = p[1]
    c_v = p[2]
    c_v3 = p[3]
    c_xv = p[4]
    c_x2v = p[5]
    c_cos = p[6]
    c_sin = p[7]
    c_xcos = p[8]
    c_vsin = p[9]
    
    # Compute features
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Time terms
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Construct prediction
    y = (c_x * x + 
         c_x3 * x3 + 
         c_v * v + 
         c_v3 * v3 + 
         c_xv * x * v + 
         c_x2v * x2 * v + 
         c_cos * cos_t + 
         c_sin * sin_t + 
         c_xcos * x * cos_t + 
         c_vsin * v * sin_t)
         
    return y

# EVOLVE-BLOCK-END
