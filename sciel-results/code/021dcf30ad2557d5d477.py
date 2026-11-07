"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a rich basis of functions.
    The evaluator fits the 10 params via BFGS.
    We provide a diverse set of basis functions that can approximate many
    common hidden functions (polynomials, trig, exponentials, sigmoids, etc.).
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Basis functions:
    # 0: constant 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: exp(-x^2)  (Gaussian bump)
    # 8: tanh(x)    (sigmoid-like)
    # 9: 1/(1+exp(-x)) (logistic, but might be ill-scaled; let's use something else)
    # Actually, let's use a mix that covers common patterns.
    # Let's stick to 10 well-behaved functions.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = x * x * x
    b4 = x * x * x * x
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-0.5 * x * x)  # Normalized-ish Gaussian
    b8 = np.tanh(x)
    b9 = np.log1p(x * x)       # Slow-growing, well-behaved on [-3,3]
    
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
