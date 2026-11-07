"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params

    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple linear prediction
        return p[0] * x + p[1]

    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t

    # Chebyshev polynomials of the first kind (orthogonal, well-conditioned)
    T0 = np.ones_like(x)
    T1 = t
    T2 = 2.0 * t2 - 1.0
    T3 = 4.0 * t3 - 3.0 * t
    T4 = 8.0 * t4 - 8.0 * t2 + 1.0
    T5 = 16.0 * t5 - 20.0 * t3 + 5.0 * t

    # Non-polynomial basis functions
    x2 = x * x
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    gauss = np.exp(-0.5 * x2)
    tanh_x = np.tanh(x)

    return (p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 +
            p[4] * T4 + p[5] * T5 +
            p[6] * sin_x + p[7] * cos_x +
            p[8] * gauss + p[9] * tanh_x)
# EVOLVE-BLOCK-END
