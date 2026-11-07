"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Center and scale inputs to [-1, 1] range
    # Domain is [0.5, 2], center is 1.25, half-width is 0.75
    x_c = (x - 1.25) / 0.75
    y_c = (y - 1.25) / 0.75
    z_c = (z - 1.25) / 0.75
    
    p = params
    
    # Strategy: Use a quadratic basis but scale the non-linear terms
    # to stabilize BFGS optimization.
    # The linear terms (indices 0-3) are unscaled.
    # The quadratic/interaction terms (indices 4-9) are scaled by 0.5.
    # This reduces the effective gradient magnitude for these terms,
    # preventing BFGS from taking large, destabilizing steps into
    # regions of high loss (which caused the score to drop to 7.4 in
    # previous un-scaled attempts).
    
    # Linear part
    pred = p[0] + p[1] * x_c + p[2] * y_c + p[3] * z_c
    
    # Quadratic part (scaled)
    # Using 0.5 scale factor
    scale = 0.5
    pred += scale * (
        p[4] * (x_c * y_c) +
        p[5] * (x_c * z_c) +
        p[6] * (y_c * z_c) +
        p[7] * (x_c * x_c) +
        p[8] * (y_c * y_c) +
        p[9] * (z_c * z_c)
    )
    
    return pred
# EVOLVE-BLOCK-END
