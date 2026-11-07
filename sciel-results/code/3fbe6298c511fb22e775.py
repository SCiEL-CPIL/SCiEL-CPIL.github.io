"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # normalize to [-1, 1]
    
    # Well-conditioned, smooth basis functions
    basis = np.column_stack([
        np.ones_like(t),
        t,
        t * t,
        t * t * t,
        np.sin(x),
        np.cos(x),
        np.sin(2.0 * x),
        np.cos(2.0 * x),
        np.exp(-t * t),
        t * np.exp(-t * t)
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
