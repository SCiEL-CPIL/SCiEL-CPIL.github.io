"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = time.time() + 145.0  # 5s safety margin


def _time_left() -> float:
    return max(0.0, _DEADLINE - time.time())


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis: constant, linear, quadratic, interaction, and trig terms.
    All terms bounded on [-1,1] for well-conditioned BFGS fitting.
    """
    c0 = params[0]   # constant
    c1 = params[1]   # x
    c2 = params[2]   # y
    c3 = params[3]   # x^2
    c4 = params[4]   # y^2
    c5 = params[5]   # x*y
    c6 = params[6]   # sin(x)
    c7 = params[7]   # sin(y)
    c8 = params[8]   # cos(x)
    c9 = params[9]   # cos(y)
    
    return (c0 + c1 * x + c2 * y + c3 * x**2 + c4 * y**2 + 
            c5 * x * y + c6 * np.sin(x) + c7 * np.sin(y) + 
            c8 * np.cos(x) + c9 * np.cos(y))
# EVOLVE-BLOCK-END
