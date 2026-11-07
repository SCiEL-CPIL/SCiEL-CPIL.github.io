"""Symbolic regression candidate: hybrid reciprocal-pairwise basis with safe division."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz
    
    This captures:
      - Linear trends (x, y, z)
      - Reciprocal/hyperbolic behavior (1/x, 1/y, 1/z)
      - Pairwise multiplicative interactions (xy, xz, yz)
      - Constant offset (1)
    
    Inputs are assumed to be in [0.5, 2] per task specification,
    but we add guards for numerical safety.
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Guard against division by zero (inputs are in [0.5, 2] so safe, but guard anyway)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    
    # Compute reciprocal terms
    inv_x = 1.0 / x_safe
    inv_y = 1.0 / y_safe
    inv_z = 1.0 / z_safe
    
    # Compute pairwise interaction terms
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Linear combination of the 10 basis terms
    result = (
        p[0] * 1.0          # constant
        + p[1] * x          # x
        + p[2] * y          # y
        + p[3] * z          # z
        + p[4] * inv_x      # 1/x
        + p[5] * inv_y      # 1/y
        + p[6] * inv_z      # 1/z
        + p[7] * xy         # x*y
        + p[8] * xz         # x*z
        + p[9] * yz         # y*z
    )
    
    return result
# EVOLVE-BLOCK-END
