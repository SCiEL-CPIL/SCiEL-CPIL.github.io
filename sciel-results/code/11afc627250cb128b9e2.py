import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (2*x - (0.5 + 2)) / (2 - 0.5) = (2*x - 2.5) / 1.5
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5
    
    # Build a basis of 10 Chebyshev polynomial terms
    # T0=1, T1=t, T2=2t^2-1, T3=4t^3-3t, etc.
    # We will use a mix of univariate and simple multivariate terms to fit within 10 params.
    # Basis:
    # 0: T0(x) * T0(y) * T0(z) = 1
    # 1: T1(x)
    # 2: T1(y)
    # 3: T1(z)
    # 4: T2(x)
    # 5: T2(y)
    # 6: T2(z)
    # 7: T1(x)*T1(y)
    # 8: T1(x)*T1(z)
    # 9: T1(y)*T1(z)
    
    T0_x = np.ones_like(tx)
    T1_x = tx
    T2_x = 2.0 * tx**2 - 1.0
    
    T0_y = np.ones_like(ty)
    T1_y = ty
    T2_y = 2.0 * ty**2 - 1.0
    
    T0_z = np.ones_like(tz)
    T1_z = tz
    T2_z = 2.0 * tz**2 - 1.0
    
    basis = np.column_stack((
        T0_x * T0_y * T0_z,
        T1_x,
        T1_y,
        T1_z,
        T2_x,
        T2_y,
        T2_z,
        T1_x * T1_y,
        T1_x * T1_z,
        T1_y * T1_z
    ))
    
    result = basis @ p
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
