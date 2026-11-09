"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear (ramp) basis in strain combined with temperature
    interactions to model elastic, yield, and hardening regimes.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # Ramp functions: max(0, strain - knot)
    # Knots chosen to capture different regimes of the stress-strain curve
    # Strain range is roughly [-0.001911, 0.9636]
    # Yield typically occurs around 0.002-0.005 for aluminium
    # Hardening continues up to ~0.5-1.0

    # Define knots for piecewise linear approximation
    knot1 = 0.002  # Near yield point
    knot2 = 0.01   # Early plastic deformation
    knot3 = 0.05   # Mid hardening
    knot4 = 0.2    # Late hardening
    knot5 = 0.5    # Near maximum strain

    # Ramp functions (differentiable everywhere, kink at knots)
    r1 = np.maximum(0.0, strain - knot1)
    r2 = np.maximum(0.0, strain - knot2)
    r3 = np.maximum(0.0, strain - knot3)
    r4 = np.maximum(0.0, strain - knot4)
    r5 = np.maximum(0.0, strain - knot5)

    # Temperature terms
    t1 = temp
    t2 = temp * temp

    # Build the model:
    # p0: intercept
    # p1: linear elastic term (slope in elastic region)
    # p2: ramp at yield (adds slope after yield)
    # p3: ramp at early plastic
    # p4: ramp at mid hardening
    # p5: ramp at late hardening
    # p6: ramp at near-max strain
    # p7: linear temperature effect
    # p8: quadratic temperature effect
    # p9: strain-temperature interaction (thermal softening/hardening)

    prediction = (
        p0 +
        p1 * strain +
        p2 * r1 +
        p3 * r2 +
        p4 * r3 +
        p5 * r4 +
        p6 * r5 +
        p7 * t1 +
        p8 * t2 +
        p9 * strain * t1
    )

    return prediction
# EVOLVE-BLOCK-END
