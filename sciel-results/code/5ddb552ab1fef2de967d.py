"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a scaled polynomial basis.
    Inputs:
        t: time
        x: position
        v: velocity
        params: 10 fitted constants
    Returns:
        Acceleration (target)
    """
    # Scale inputs to O(1) range to improve BFGS conditioning
    # t in [30, 50] -> t_s in [-1, 1]
    t_s = (t - 40.0) / 10.0
    # x in [-0.2, 0.17] -> x_s in [-1, 0.85] approx
    x_s = x / 0.2
    # v in [-0.34, 0.36] -> v_s in [-1.7, 1.8] approx
    v_s = v / 0.2

    # Define basis functions
    f0 = np.ones_like(t_s)
    f1 = t_s
    f2 = x_s
    f3 = v_s
    f4 = x_s**2
    f5 = x_s**3
    f6 = v_s**2
    f7 = x_s * v_s
    f8 = x_s**2 * v_s
    f9 = v_s**3

    # Combine with params
    return (params[0] * f0 + params[1] * f1 + params[2] * f2 + params[3] * f3 +
            params[4] * f4 + params[5] * f5 + params[6] * f6 + params[7] * f7 +
            params[8] * f8 + params[9] * f9)
# EVOLVE-BLOCK-END
