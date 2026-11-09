import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: hybrid basis combining normalized linear terms
    with log-space quadratic interactions.
    
    This model class captures both additive linear relationships (via normalized
    inputs u, v, w) and multiplicative/power-law structures (via log-space
    interactions). The 10 parameters are allocated as:
    - 1 constant
    - 3 linear terms in normalized coordinates (u, v, w)
    - 3 linear terms in log-space (lx, ly, lz)
    - 3 interaction terms in log-space (lx*ly, lx*lz, ly*lz)
    
    Normalization: inputs are in [0.5, 2.0], so center=1.25, half_range=0.75.
    This improves conditioning for BFGS optimization.
    """
    p = params
    
    # Normalize to [-1, 1] for better numerical conditioning
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Log-transformed inputs (safe since inputs in [0.5, 2])
    x_safe = np.clip(x, 1e-8, None)
    y_safe = np.clip(y, 1e-8, None)
    z_safe = np.clip(z, 1e-8, None)
    
    lx = np.log(x_safe)
    ly = np.log(y_safe)
    lz = np.log(z_safe)
    
    # Hybrid basis with 10 coefficients:
    # p[0]: constant
    # p[1]: u (normalized x)
    # p[2]: v (normalized y)
    # p[3]: w (normalized z)
    # p[4]: lx (log x)
    # p[5]: ly (log y)
    # p[6]: lz (log z)
    # p[7]: lx * ly (log interaction x,y)
    # p[8]: lx * lz (log interaction x,z)
    # p[9]: ly * lz (log interaction y,z)
    result = (p[0] + 
              p[1]*u + p[2]*v + p[3]*w +
              p[4]*lx + p[5]*ly + p[6]*lz +
              p[7]*lx*ly + p[8]*lx*lz + p[9]*ly*lz)
    
    return result
