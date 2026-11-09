"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining linear, quadratic (centered), and non-linear basis functions.
    
    The linear model scored 1.0 (predicting mean), suggesting the true function is non-linear.
    The full quadratic model scored 7.39, which is worse than linear, likely due to:
    1. BFGS getting stuck in a poor local minimum starting from all ones.
    2. Ill-conditioned basis when using raw inputs.
    
    Strategy:
    - Center inputs to improve conditioning.
    - Use a mix of polynomial and non-polynomial terms to cover a broader function space.
    - Keep the model simple enough for BFGS to converge reliably.
    
    Basis (10 params):
    p0: 1 (bias)
    p1: x_c (centered x)
    p2: y_c (centered y)
    p3: z_c (centered z)
    p4: x_c^2
    p5: y_c^2
    p6: z_c^2
    p7: x_c * y_c
    p8: x_c * z_c
    p9: y_c * z_c
    
    This is the centered quadratic model. Centering should help BFGS converge better
    by reducing the correlation between linear and quadratic terms.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Center the inputs around the midpoint of [0.5, 2], which is 1.25
    x_c = x - 1.25
    y_c = y - 1.25
    z_c = z - 1.25
    
    p = params
    
    # Build the prediction as a linear combination of centered quadratic basis functions
    pred = (p[0] * np.ones_like(x) +
            p[1] * x_c +
            p[2] * y_c +
            p[3] * z_c +
            p[4] * x_c**2 +
            p[5] * y_c**2 +
            p[6] * z_c**2 +
            p[7] * x_c * y_c +
            p[8] * x_c * z_c +
            p[9] * y_c * z_c)
    
    return pred
# EVOLVE-BLOCK-END
