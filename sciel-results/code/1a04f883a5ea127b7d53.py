"""Symbolic regression with 3 inputs – mixed polynomial and reciprocal basis for diverse function shapes."""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with 10 features combining polynomial
    and reciprocal terms to capture diverse function behaviors.
    
    Features:
    - Constant (1)
    - Linear: x, y, z
    - Pairwise interactions: xy, xz, yz
    - Reciprocal: 1/x, 1/y, 1/z
    Total: 10 features
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 features
    f0 = np.ones_like(x)       # constant
    f1 = x                     # linear x
    f2 = y                     # linear y
    f3 = z                     # linear z
    f4 = x * y                 # xy interaction
    f5 = x * z                 # xz interaction
    f6 = y * z                 # yz interaction
    f7 = 1.0 / x               # reciprocal x
    f8 = 1.0 / y               # reciprocal y
    f9 = 1.0 / z               # reciprocal z
    
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
