import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Fourier basis functions.
    Maps x in [-2, 2] to t in [-pi, pi].
    Uses 10 parameters for 1, sin(t), cos(t), sin(2t), cos(2t), sin(3t), cos(3t), sin(4t), cos(4t), sin(5t).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure x is 1D for consistent basis construction
    if x.ndim != 1:
        x = x.flatten()
        
    N = x.shape[0]
    
    # Map x from [-2, 2] to t in [-pi, pi]
    t = np.pi / 2.0 * x
    
    # Construct basis matrix
    # We use 10 basis functions:
    # 0: 1
    # 1: sin(t)
    # 2: cos(t)
    # 3: sin(2t)
    # 4: cos(2t)
    # 5: sin(3t)
    # 6: cos(3t)
    # 7: sin(4t)
    # 8: cos(4t)
    # 9: sin(5t)
    
    basis = np.empty((N, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = np.sin(t)
    basis[:, 2] = np.cos(t)
    basis[:, 3] = np.sin(2.0 * t)
    basis[:, 4] = np.cos(2.0 * t)
    basis[:, 5] = np.sin(3.0 * t)
    basis[:, 6] = np.cos(3.0 * t)
    basis[:, 7] = np.sin(4.0 * t)
    basis[:, 8] = np.cos(4.0 * t)
    basis[:, 9] = np.sin(5.0 * t)
    
    # Linear combination
    return basis @ params
