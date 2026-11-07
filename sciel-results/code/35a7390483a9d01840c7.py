"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature using a high-order separable polynomial."""
    
    # Normalization constants derived from the training data ranges provided in the task contract
    e_min = -0.001911
    e_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    e_range = e_max - e_min
    t_range = t_max - t_min
    
    # Ensure ranges are non-zero
    if e_range < 1e-12:
        e_range = 1.0
    if t_range < 1e-12:
        t_range = 1.0
        
    # Normalize inputs to [0, 1]
    u = (strain - e_min) / e_range
    v = (temp - t_min) / t_range
    
    # Clip to [0, 1] to ensure stability if test points are slightly out of bounds
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Pre-compute powers
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    p = params
    
    # Basis functions: Separable 4th order in u and v, plus linear interaction
    # Terms: 1, u, u^2, u^3, u^4, v, v^2, v^3, v^4, uv
    stress = (
        p[0] * 1.0 +
        p[1] * u +
        p[2] * u2 +
        p[3] * u3 +
        p[4] * u4 +
        p[5] * v +
        p[6] * v2 +
        p[7] * v3 +
        p[8] * v4 +
        p[9] * (u * v)
    )
    
    return stress
# EVOLVE-BLOCK-END
