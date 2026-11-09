"""Symbolic regression using a Chebyshev polynomial basis of degree 9."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using Chebyshev polynomials of the first kind, degree 0..9.
    The input x is in [-2, 2]; we scale to t in [-1, 1] via t = x/2
    so that Chebyshev polynomials are well-conditioned on their natural domain.

    params (length 10): coefficients c0..c9 for T_0(t), T_1(t), ..., T_9(t).

    The function is:
        f(x) = sum_{k=0}^{9} c_k * T_k(x/2)

    where T_k is the k-th Chebyshev polynomial of the first kind.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 2.0  # scale to [-1, 1]

    # Compute Chebyshev polynomials T_0 through T_9 via recurrence:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7

    result = (
        params[0] * T0
        + params[1] * T1
        + params[2] * T2
        + params[3] * T3
        + params[4] * T4
        + params[5] * T5
        + params[6] * T6
        + params[7] * T7
        + params[8] * T8
        + params[9] * T9
    )

    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
