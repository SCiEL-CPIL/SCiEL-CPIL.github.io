"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # 10 diverse basis functions on x in [-2, 2]
    # p0: constant
    # p1: x (linear)
    # p2: x^2 (quadratic)
    # p3: x^3 (cubic)
    # p4: x^4 (quartic)
    # p5: sin(x)
    # p6: cos(x)
    # p7: exp(x)
    # p8: exp(-x)
    # p9: tanh(x)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    y = np.full_like(x, p0, dtype=np.float64)
    y += p1 * x
    y += p2 * x2
    y += p3 * x3
    y += p4 * x4
    y += p5 * np.sin(x)
    y += p6 * np.cos(x)
    y += p7 * np.exp(x)
    y += p8 * np.exp(-x)
    y += p9 * np.tanh(x)
    
    return y
# EVOLVE-BLOCK-END
