import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Fourier Basis on [-3, 3]
    # Frequencies chosen to cover the range with periods 6, 3, 2, 1.5, 1.2
    # k=1: period 6
    # k=2: period 3
    # k=3: period 2
    # k=4: period 1.5
    # k=5: period 1.2
    
    B = np.ones((x.shape[0], 10))
    
    # p0: DC
    # p1, p2: k=1
    B[:, 1] = np.sin(np.pi * x / 3.0)
    B[:, 2] = np.cos(np.pi * x / 3.0)
    
    # p3, p4: k=2
    B[:, 3] = np.sin(2.0 * np.pi * x / 3.0)
    B[:, 4] = np.cos(2.0 * np.pi * x / 3.0)
    
    # p5, p6: k=3
    B[:, 5] = np.sin(3.0 * np.pi * x / 3.0)
    B[:, 6] = np.cos(3.0 * np.pi * x / 3.0)
    
    # p7, p8: k=4
    B[:, 7] = np.sin(4.0 * np.pi * x / 3.0)
    B[:, 8] = np.cos(4.0 * np.pi * x / 3.0)
    
    # p9: k=5 (Sine only to keep param count at 10)
    B[:, 9] = np.sin(5.0 * np.pi * x / 3.0)
    
    result = B @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
