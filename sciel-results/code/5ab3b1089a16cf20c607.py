"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of scaled basis functions for symbolic regression.
    Basis functions are scaled to have similar magnitudes to improve
    numerical conditioning for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Precompute common values
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x
    
    # Scaled basis functions (each roughly O(1) in magnitude over [0.1, 6])
    # b0: constant
    b0 = np.ones_like(x)
    # b1: linear
    b1 = x / 3.0  # scale to ~[-0.03, 2]
    # b2: quadratic
    b2 = x2 / 9.0  # scale to ~[0.001, 4]
    # b3: cubic
    b3 = x3 / 27.0  # scale to ~[0.0001, 8]
    # b4: quartic
    b4 = x4 / 81.0  # scale to ~[0.00001, 16]
    # b5: quintic
    b5 = x5 / 243.0  # scale to ~[0.000001, 32]
    # b6: exponential (scaled)
    b6 = np.exp(x) / 100.0  # exp(6)~403, so /100 gives ~[0.01, 4]
    # b7: exp(-x)
    b7 = np.exp(-x) * 10.0  # exp(-6)~0.0025, so *10 gives ~[0.025, 10]
    # b8: log
    b8 = np.log(x) * 2.0  # log(0.1)~-2.3, log(6)~1.8, so *2 gives ~[-4.6, 3.6]
    # b9: reciprocal
    b9 = (1.0 / x) * 0.5  # 1/0.1=10, 1/6~0.17, so *0.5 gives ~[0.08, 5]
    
    # Linear combination
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return y

# EVOLVE-BLOCK-END
