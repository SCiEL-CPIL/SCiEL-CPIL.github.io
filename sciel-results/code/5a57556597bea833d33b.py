# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Linear model in a hand-crafted feature basis of 10 functions.
    # This is convex in params, so BFGS should converge reliably.
    # Features chosen to span common symbolic regression patterns:
    # constants, linears, pairwise products, triple product, and a ratio.
    
    # Build feature matrix: shape (n, 10)
    # f1 = 1 (bias)
    # f2 = x
    # f3 = y
    # f4 = z
    # f5 = x*y
    # f6 = x*z
    # f7 = y*z
    # f8 = x*y*z
    # f9 = x**2
    # f10 = 1/z
    
    n = len(x)
    F = np.empty((n, 10), dtype=np.float64)
    F[:, 0] = 1.0
    F[:, 1] = x
    F[:, 2] = y
    F[:, 3] = z
    F[:, 4] = x * y
    F[:, 5] = x * z
    F[:, 6] = y * z
    F[:, 7] = x * y * z
    F[:, 8] = x * x
    F[:, 9] = 1.0 / z
    
    return F @ params
# EVOLVE-BLOCK-END
