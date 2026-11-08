"""Nonlinear oscillator symbolic regression model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator
    with time-dependent driving force.
    
    Uses up to 10 fitted constants. Structure:
    - Time-dependent drive: polynomial in centered/scaled time
    - State-dependent terms: polynomial in x and v (Duffing-like)
    
    Inputs:
        t: time array, range [30, 50]
        x: position array, range [-0.2056, 0.1749]
        v: velocity array, range [-0.3391, 0.3624]
        params: array of 10 fitted constants
    """
    # Center and scale time for better conditioning
    # t range [30, 50], center at 40, scale by 10
    tc = (t - 40.0) / 10.0
    tc2 = tc * tc
    tc3 = tc2 * tc

    # Scale x and v to unit-ish range for conditioning
    # x range ~[-0.21, 0.17], center ~-0.02, scale ~0.2
    xc = (x - (-0.015)) / 0.2
    # v range ~[-0.34, 0.36], center ~0.01, scale ~0.35
    vc = (v - 0.01) / 0.35

    # Powers of scaled variables
    xc2 = xc * xc
    xc3 = xc2 * xc
    vc2 = vc * vc
    vc3 = vc2 * vc

    # Cross terms
    xcv = xc * vc
    xcv2 = xcv * vc
    xc2v = xc2 * vc

    # Build the expression with 10 parameters
    # p0: bias/constant
    # p1: linear time drive
    # p2: quadratic time drive
    # p3: cubic time drive
    # p4: linear x (restoring force)
    # p5: cubic x (Duffing nonlinearity)
    # p6: linear v (damping)
    # p7: quadratic v
    # p8: x*v cross term
    # p9: x^2 * v cross term

    result = (
        params[0]
        + params[1] * tc
        + params[2] * tc2
        + params[3] * tc3
        + params[4] * xc
        + params[5] * xc3
        + params[6] * vc
        + params[7] * vc2
        + params[8] * xcv
        + params[9] * xc2v
    )

    return result
# EVOLVE-BLOCK-END
