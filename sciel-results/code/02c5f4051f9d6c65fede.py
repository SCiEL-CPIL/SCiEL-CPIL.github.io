"""Symbolic regression with hybrid polynomial and trigonometric basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    Uses a hybrid basis of linear, interaction, and sinusoidal terms.
    This basis was observed to outperform pure polynomial bases in previous evaluations.
    
    Basis (10 terms):
    0: 1
    1: x
    2: y
    3: z
    4: x*y
    5: x*z
    6: y*z
    7: sin(x)
    8: sin(y)
    9: sin(z)
    """
    p = params
    return (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * y +
        p[3] * z +
        p[4] * x * y +
        p[5] * x * z +
        p[6] * y * z +
        p[7] * np.sin(x) +
        p[8] * np.sin(y) +
        p[9] * np.sin(z)
    )
# EVOLVE-BLOCK-END
