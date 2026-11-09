"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Strategy: Use a quadratic polynomial basis in log-space.
    # This model class captures power-law relationships (Y ~ x^a * y^b * z^c)
    # and multiplicative structures much better than additive polynomials.
    # Inputs x, y, z are in [0.5, 2.0], so log is well-defined and stable.
    
    # Clip to ensure positive inputs for log (though range is already positive)
    x_safe = np.clip(x, 1e-8, None)
    y_safe = np.clip(y, 1e-8, None)
    z_safe = np.clip(z, 1e-8, None)
    
    lx = np.log(x_safe)
    ly = np.log(y_safe)
    lz = np.log(z_safe)
    
    p = params
    
    # Basis:
    # 0: 1
    # 1: lx
    # 2: ly
    # 3: lz
    # 4: lx^2
    # 5: ly^2
    # 6: lz^2
    # 7: lx*ly
    # 8: lx*lz
    # 9: ly*lz
    
    return (p[0] + 
            p[1] * lx + 
            p[2] * ly + 
            p[3] * lz + 
            p[4] * lx * lx + 
            p[5] * ly * ly + 
            p[6] * lz * lz + 
            p[7] * lx * ly + 
            p[8] * lx * lz + 
            p[9] * ly * lz)
# EVOLVE-BLOCK-END
