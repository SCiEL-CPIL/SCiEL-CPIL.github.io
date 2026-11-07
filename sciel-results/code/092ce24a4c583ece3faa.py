# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with a mixed polynomial-reciprocal-ratio basis.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: z
    4: 1/x
    5: 1/y
    6: 1/z
    7: x/y
    8: y/z
    9: z/x
    
    This basis is designed to capture rational-like structures and
    multiplicative interactions that neither pure polynomials nor
    trigonometric functions can approximate well. The ratio terms
    (x/y, y/z, z/x) and reciprocal terms (1/x, 1/y, 1/z) allow the
    linear combination to approximate functions with denominators.
    """
    p = params
    
    b0 = 1.0
    b1 = x
    b2 = y
    b3 = z
    b4 = 1.0 / x
    b5 = 1.0 / y
    b6 = 1.0 / z
    b7 = x / y
    b8 = y / z
    b9 = z / x
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
