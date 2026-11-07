# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters for symbolic regression.
    
    Approach: Rational/Reciprocal Basis.
    This basis includes constant, linear, pairwise product, and reciprocal terms.
    This differs from the previous polynomial-only approach by allowing
    hyperbolic behavior (e.g., 1/x) and ensuring a bias term is present.
    
    Basis functions (10 total):
    0: 1.0 (Bias)
    1: x
    2: y
    3: z
    4: x * y
    5: y * z
    6: z * x
    7: 1.0 / x
    8: 1.0 / y
    9: 1.0 / z
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Build prediction from basis
    # Note: x, y, z are >= 0.5, so 1/x is safe and bounded in [0.5, 2]
    pred = (
        p0 * 1.0 +
        p1 * x +
        p2 * y +
        p3 * z +
        p4 * x * y +
        p5 * y * z +
        p6 * z * x +
        p7 * (1.0 / x) +
        p8 * (1.0 / y) +
        p9 * (1.0 / z)
    )
    
    return pred
# EVOLVE-BLOCK-END
