import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (2*v - (a+b)) / (b-a) where a=0.5, b=2
    # t = (2*v - 2.5) / 1.5 = (2*v - 2.5) / 1.5
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5
    
    # Clip to [-1, 1] for safety (should already be in range, but numerical safety)
    tx = np.clip(tx, -1.0, 1.0)
    ty = np.clip(ty, -1.0, 1.0)
    tz = np.clip(tz, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n(t) using recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    def chebyshev(t):
        """Return T_0, T_1, T_2, T_3 for input t"""
        T0 = np.ones_like(t)
        T1 = t
        T2 = 2.0 * t * T1 - T0
        T3 = 2.0 * t * T2 - T1
        return T0, T1, T2, T3
    
    Tx0, Tx1, Tx2, Tx3 = chebyshev(tx)
    Ty0, Ty1, Ty2, Ty3 = chebyshev(ty)
    Tz0, Tz1, Tz2, Tz3 = chebyshev(tz)
    
    # Build 10 basis functions using Chebyshev polynomials
    # Strategy: mix univariate and interaction terms
    # 0: T0(x)*T0(y)*T0(z) = 1 (constant)
    # 1: T1(x) (linear x)
    # 2: T1(y) (linear y)
    # 3: T1(z) (linear z)
    # 4: T2(x) (quadratic x)
    # 5: T2(y) (quadratic y)
    # 6: T2(z) (quadratic z)
    # 7: T1(x)*T1(y) (interaction xy)
    # 8: T1(x)*T1(z) (interaction xz)
    # 9: T1(y)*T1(z) (interaction yz)
    
    b0 = Tx0 * Ty0 * Tz0
    b1 = Tx1
    b2 = Ty1
    b3 = Tz1
    b4 = Tx2
    b5 = Ty2
    b6 = Tz2
    b7 = Tx1 * Ty1
    b8 = Tx1 * Tz1
    b9 = Ty1 * Tz1
    
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
