"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Use a well-conditioned basis of 10 features linear in params.
# Key improvements over previous version:
# - Better conditioned features (less correlation)
# - tanh terms for bounded non-linear behavior
# - log terms (safe on [0.5, 2]) for smooth non-polynomial coverage
# - Centered and scaled features for BFGS stability

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 diverse, well-conditioned basis features.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Centered variables (mean of [0.5, 2] is 1.25)
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25
    
    # Build 10 diverse, well-conditioned features
    f0 = np.ones_like(x)                    # constant
    f1 = cx                                 # linear x
    f2 = cy                                 # linear y
    f3 = cz                                 # linear z
    f4 = cx * cy                            # xy interaction
    f5 = cx * cz                            # xz interaction
    f6 = cy * cz                            # yz interaction
    f7 = np.tanh(cx + cy + cz)             # bounded non-linear (sum)
    f8 = np.log(x) + np.log(y) + np.log(z) # log terms (smooth on [0.5,2])
    f9 = cx * cy * cz                       # three-way interaction
    
    # Linear combination
    pred = (params[0] * f0 + 
            params[1] * f1 + 
            params[2] * f2 + 
            params[3] * f3 + 
            params[4] * f4 + 
            params[5] * f5 + 
            params[6] * f6 + 
            params[7] * f7 + 
            params[8] * f8 + 
            params[9] * f9)
    
    return pred
# EVOLVE-BLOCK-END
