import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Tensor-product Fourier basis with linear terms for fast convergence
    # First 3 terms: constant, linear-x, linear-y (captures dominant signal)
    # Remaining 7 terms: Fourier modes + one interaction
    c0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = np.sin(np.pi * x)
    t4 = np.sin(np.pi * y)
    t5 = np.cos(2.0 * np.pi * x)
    t6 = np.cos(2.0 * np.pi * y)
    t7 = np.sin(2.0 * np.pi * x)
    t8 = np.sin(2.0 * np.pi * y)
    t9 = x * y
    
    pred = (params[0] * c0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = params[0] + params[1] * x + params[2] * y
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
            
    return pred
