import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: Linear terms for global trend, Trig for local/periodic features
    # This addresses the boundary mismatch of pure trig bases while retaining
    # their strength in capturing oscillatory behavior.
    
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    
    xy = x * y
    sin_px_sin_py = sin_px * sin_py
    cos_px_cos_py = cos_px * cos_py
    
    # Basis vector
    # 0: 1
    # 1: x
    # 2: y
    # 3: sin(pi x)
    # 4: sin(pi y)
    # 5: cos(pi x)
    # 6: cos(pi y)
    # 7: x * y
    # 8: sin(pi x) * sin(pi y)
    # 9: cos(pi x) * cos(pi y)
    
    f = (params[0] * 1 +
         params[1] * x +
         params[2] * y +
         params[3] * sin_px +
         params[4] * sin_py +
         params[5] * cos_px +
         params[6] * cos_py +
         params[7] * xy +
         params[8] * sin_px_sin_py +
         params[9] * cos_px_cos_py)
         
    return f
