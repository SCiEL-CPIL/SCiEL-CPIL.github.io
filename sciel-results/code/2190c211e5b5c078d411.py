"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses 10 basis features to span common nonlinear oscillator dynamics:
    constant, t, x, v, x^2, x^3, v^2, x*v, sin(t), cos(t)
    """
    return (
        params[0]
        + params[1] * t
        + params[2] * x
        + params[3] * v
        + params[4] * x * x
        + params[5] * x * x * x
        + params[6] * v * v
        + params[7] * x * v
        + params[8] * np.sin(t)
        + params[9] * np.cos(t)
    )
# EVOLVE-BLOCK-END
