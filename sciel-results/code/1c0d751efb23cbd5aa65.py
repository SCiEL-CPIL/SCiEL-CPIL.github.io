import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Structural change: use a mixed basis with polynomial, log, and inverse terms
    # to capture rational/logarithmic function behavior that pure polynomials miss
    
    # Safe logs and inverses (inputs are in [0.5, 2], so all positive)
    log_x = np.log(x)
    log_y = np.log(y)
    log_z = np.log(z)
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    # Build 10 basis functions:
    # 0: constant
    # 1: x
    # 2: y
    # 3: z
    # 4: log(x)
    # 5: log(y)
    # 6: log(z)
    # 7: 1/x
    # 8: 1/y
    # 9: 1/z
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = log_x
    b5 = log_y
    b6 = log_z
    b7 = inv_x
    b8 = inv_y
    b9 = inv_z
    
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
