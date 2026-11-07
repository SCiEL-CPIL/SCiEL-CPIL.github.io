import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with physically-motivated basis,
    normalized features for BFGS conditioning, and a small number of
    higher-order interaction terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize x and v to roughly unit scale to improve Hessian conditioning
    # x in [-0.3391, 0.3624], v in [-0.2056, 0.1749]
    xs = x / 0.35
    vs = v / 0.20

    # Build basis
    # b0: constant
    # b1: xs (linear restoring)
    # b2: xs^3 (Duffing nonlinear restoring)
    # b3: vs (linear damping)
    # b4: xs * vs (parametric coupling)
    # b5: sin(t) (forcing in phase)
    # b6: cos(t) (forcing quadrature)
    # b7: xs * sin(t) (state-modulated forcing)
    # b8: vs * cos(t) (velocity-modulated forcing)
    # b9: xs^2 * vs (higher-order damping/restoring interaction)

    b0 = np.ones_like(t, dtype=np.float64)
    b1 = xs
    b2 = xs * xs * xs
    b3 = vs
    b4 = xs * vs
    b5 = np.sin(t)
    b6 = np.cos(t)
    b7 = xs * np.sin(t)
    b8 = vs * np.cos(t)
    b9 = xs * xs * vs

    result = (p0 * b0 +
              p1 * b1 +
              p2 * b2 +
              p3 * b3 +
              p4 * b4 +
              p5 * b5 +
              p6 * b6 +
              p7 * b7 +
              p8 * b8 +
              p9 * b9)

    # Replace any non-finite values with 0 to avoid optimizer failure
    result = np.where(np.isfinite(result), result, 0.0)

    return result
