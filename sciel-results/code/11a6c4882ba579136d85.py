"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two low-degree polynomials.
    
    The function is modeled as P_4(t) * Q_4(t), where t = x/2 maps the input
    from [-2, 2] to [-1, 1] for numerical stability.
    
    P_4(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
    Q_4(t) = b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4
    
    The 10 parameters are [a0, a1, a2, a3, a4, b0, b1, b2, b3, b4].
    This non-convex but structured landscape can capture sharp local features
    and high-frequency oscillations better than linear basis combinations.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Split parameters
    a = p[:5]
    b = p[5:10]
    
    # Compute powers of t
    t0 = np.ones_like(t)
    t1 = t
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    # Compute P_4(t)
    P = a[0] * t0 + a[1] * t1 + a[2] * t2 + a[3] * t3 + a[4] * t4
    
    # Compute Q_4(t)
    Q = b[0] * t0 + b[1] * t1 + b[2] * t2 + b[3] * t3 + b[4] * t4
    
    return P * Q
# EVOLVE-BLOCK-END
