import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has at least 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    params = params[:10]
    
    # Basis functions:
    # 0: Constant
    # 1: Linear x
    # 2: Linear y
    # 3: Quadratic x^2
    # 4: Quadratic y^2
    # 5: Interaction xy
    # 6: Sine of x (scaled to pi to cover period 2)
    # 7: Sine of y
    # 8: Cosine of x
    # 9: Cosine of y
    
    term_1 = np.ones_like(x)
    term_2 = x
    term_3 = y
    term_4 = x**2
    term_5 = y**2
    term_6 = x * y
    term_7 = np.sin(np.pi * x)
    term_8 = np.sin(np.pi * y)
    term_9 = np.cos(np.pi * x)
    term_10 = np.cos(np.pi * y)
    
    result = (params[0] * term_1 +
              params[1] * term_2 +
              params[2] * term_3 +
              params[3] * term_4 +
              params[4] * term_5 +
              params[5] * term_6 +
              params[6] * term_7 +
              params[7] * term_8 +
              params[8] * term_9 +
              params[9] * term_10)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
