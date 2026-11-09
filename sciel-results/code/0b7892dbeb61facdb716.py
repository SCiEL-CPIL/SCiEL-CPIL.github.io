import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    xm = (2.0 * x - 2.5) / 1.5
    ym = (2.0 * y - 2.5) / 1.5
    zm = (2.0 * z - 2.5) / 1.5
    
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(xm)
    T1_x = xm
    T2_x = 2.0 * xm * xm - 1.0
    
    T0_y = np.ones_like(ym)
    T1_y = ym
    T2_y = 2.0 * ym * ym - 1.0
    
    T0_z = np.ones_like(zm)
    T1_z = zm
    T2_z = 2.0 * zm * zm - 1.0
    
    # Basis: 10 terms, linear in params
    basis0 = T0_x * T0_y * T0_z
    basis1 = T1_x * T0_y * T0_z
    basis2 = T0_x * T1_y * T0_z
    basis3 = T0_x * T0_y * T1_z
    basis4 = T2_x * T0_y * T0_z
    basis5 = T0_x * T2_y * T0_z
    basis6 = T0_x * T0_y * T2_z
    basis7 = T1_x * T1_y * T0_z
    basis8 = T1_x * T0_y * T1_z
    basis9 = T0_x * T1_y * T1_z
    
    result = (p[0] * basis0 + p[1] * basis1 + p[2] * basis2 + p[3] * basis3 +
              p[4] * basis4 + p[5] * basis5 + p[6] * basis6 + p[7] * basis7 +
              p[8] * basis8 + p[9] * basis9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
