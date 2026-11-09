"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: combines linear terms in normalized inputs, log-quadratic terms,
    and ratio-based features to capture additive, power-law, and interaction structures.
    
    10 parameters allocated as:
    - 3: linear in u, v, w (normalized inputs)
    - 1: constant
    - 3: linear in log-space (lx, ly, lz)
    - 3: log-quadratic cross terms (lx*ly, lx*lz, ly*lz)
    """
    p = params
    
    # Normalize inputs to [-1, 1] for better conditioning
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Log-transformed inputs (safe since inputs in [0.5, 2])
    lx = np.log(x)
    ly = np.log(y)
    lz = np.log(z)
    
    # Basis terms:
    # 0: constant
    # 1: u (linear x)
    # 2: v (linear y)
    # 3: w (linear z)
    # 4: lx (log x)
    # 5: ly (log y)
    # 6: lz (log z)
    # 7: lx * ly (interaction in log-space)
    # 8: lx * lz (interaction in log-space)
    # 9: ly * lz (interaction in log-space)
    
    return (p[0] +
            p[1] * u +
            p[2] * v +
            p[3] * w +
            p[4] * lx +
            p[5] * ly +
            p[6] * lz +
            p[7] * lx * ly +
            p[8] * lx * lz +
            p[9] * ly * lz)
# EVOLVE-BLOCK-END
