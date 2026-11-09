"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a scaled cubic polynomial basis.
    Scaling inputs to O(1) range improves conditioning for BFGS optimization.
    """
    # Scale inputs to approximately [-1, 1] range to fix ill-conditioning
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    xs = x / 0.6
    vs = v / 0.3
    
    # Compute powers
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    
    # Basis terms
    # 1. Constant
    term0 = np.ones_like(x)
    # 2. Linear x
    term1 = xs
    # 3. Linear v
    term2 = vs
    # 4. Quadratic x
    term3 = xs2
    # 5. Cross term xv
    term4 = xs * vs
    # 6. Quadratic v
    term5 = vs2
    # 7. Cubic x
    term6 = xs3
    # 8. Cross term x^2v
    term7 = xs2 * vs
    # 9. Cross term xv^2
    term8 = xs * vs2
    # 10. Cubic v
    term9 = vs3
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
