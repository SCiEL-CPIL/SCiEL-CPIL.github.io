"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# The evaluator calls equation(t, x, v, params).
# Nonlinear oscillator acceleration model:
# y = driving(t) + spring(x) + damping(x, v)
# with 10 fitted constants total.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.

    y = p0*sin(t) + p1*cos(t)
        + p2*x + p3*x^2 + p4*x^3
        + p5*v + p6*x*v + p7*x^2*v
        + p8*v^2 + p9*v^3
    """
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v

    y = (params[0] * sin_t
         + params[1] * cos_t
         + params[2] * x
         + params[3] * x2
         + params[4] * x3
         + params[5] * v
         + params[6] * xv
         + params[7] * x2v
         + params[8] * v2
         + params[9] * v3)

    return y
# EVOLVE-BLOCK-END
