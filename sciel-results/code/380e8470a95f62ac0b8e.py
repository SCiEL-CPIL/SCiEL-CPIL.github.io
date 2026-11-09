"""Symbolic regression with 3 inputs – shifted Legendre polynomial basis for orthonormal conditioning."""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 features based on shifted Legendre
    polynomials (orthonormal over [0.5, 2]) for good BFGS conditioning.
    
    Features:
    - P0, P1, P2 for x (3 features)
    - P0, P1, P2 for y (3 features)
    - P0, P1, P2 for z (3 features)
    - P1(x)*P1(y) interaction (1 feature)
    Total: 10 features
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map [0.5, 2] to [-1, 1] for standard Legendre polynomials
    # t = (4*v - 5) / 3
    tx = (4.0 * x - 5.0) / 3.0
    ty = (4.0 * y - 5.0) / 3.0
    tz = (4.0 * z - 5.0) / 3.0
    
    # Legendre polynomials: P0=1, P1=t, P2=(3t^2-1)/2
    P0x = np.ones_like(tx)
    P1x = tx
    P2x = 0.5 * (3.0 * tx * tx - 1.0)
    
    P0y = np.ones_like(ty)
    P1y = ty
    P2y = 0.5 * (3.0 * ty * ty - 1.0)
    
    P0z = np.ones_like(tz)
    P1z = tz
    P2z = 0.5 * (3.0 * tz * tz - 1.0)
    
    # 10 features
    f0 = P0x          # constant (same as P0y, P0z)
    f1 = P1x          # linear in x
    f2 = P2x          # quadratic in x
    f3 = P1y          # linear in y
    f4 = P2y          # quadratic in y
    f5 = P1z          # linear in z
    f6 = P2z          # quadratic in z
    f7 = P1x * P1y    # xy interaction
    f8 = P1x * P1z    # xz interaction
    f9 = P1y * P1z    # yz interaction
    
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
