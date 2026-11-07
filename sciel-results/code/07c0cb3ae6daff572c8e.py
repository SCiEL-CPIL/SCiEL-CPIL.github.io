# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Polynomial-Trigonometric Basis.
    
    The current program uses a purely trigonometric basis (score ~4.59).
    The grand-parent used a quadratic polynomial basis (score ~7.39).
    The best inspiration (score ~1.30) used a mix of linear, sin, and exp.
    
    This program adopts a hybrid basis that includes:
    1. Constant
    2-4. Linear terms (x, y, z)
    5-7. Quadratic terms (x^2, y^2, z^2)
    8-10. Fundamental sine terms (sin(pi*x), sin(pi*y), sin(pi*z))
    
    This structure allows the model to capture both smooth polynomial trends
    and oscillatory behaviors simultaneously. The basis functions are linear
    in the parameters, ensuring BFGS optimization is convex and robust.
    """
    p = params
    
    # Polynomial components
    x_sq = x * x
    y_sq = y * y
    z_sq = z * z
    
    # Trigonometric components (fundamental frequency for domain [0.5, 2])
    # pi * x maps [0.5, 2] to [pi/2, 2pi], covering a full half-wave to zero-crossing
    sin_pi_x = np.sin(np.pi * x)
    sin_pi_y = np.sin(np.pi * y)
    sin_pi_z = np.sin(np.pi * z)
    
    return (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * z
            + p[4] * x_sq
            + p[5] * y_sq
            + p[6] * z_sq
            + p[7] * sin_pi_x
            + p[8] * sin_pi_y
            + p[9] * sin_pi_z)
# EVOLVE-BLOCK-END
