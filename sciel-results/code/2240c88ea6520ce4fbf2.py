"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    Assumes autonomous dynamics (Van der Pol / Duffing type).
    Uses up to 10 fitted constants.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # 1. Linear restoring force
    term1 = p0 * x

    # 2. Cubic restoring force (Duffing)
    term2 = p1 * x**3

    # 3. Linear damping
    term3 = p2 * v

    # 4. Van der Pol style damping (negative damping for small x, positive for large x)
    # Usually modeled as -mu * x^2 * v, so here p3 * x^2 * v
    term4 = p3 * x**2 * v

    # 5. Cubic damping (velocity saturation)
    term5 = p4 * v**3

    # 6. Cross term x*v
    term6 = p5 * x * v

    # 7. Position squared term
    term7 = p6 * x**2

    # 8. Velocity squared term (quadratic drag)
    term8 = p7 * v**2

    # 9. Mixed term x * v^2
    term9 = p8 * x * v**2

    # 10. Constant offset
    term10 = p9

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
# EVOLVE-BLOCK-END
