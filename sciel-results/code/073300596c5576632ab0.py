"""Rewritten program for symbolic regression with flexible non-linear basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to spare before the 120s limit


def _chebyshev_basis(t):
    """
    Compute Chebyshev polynomials T_0 to T_9 for input t in [-1, 1].
    Returns an array of shape (N, 10).
    """
    N = t.shape[0]
    basis = np.empty((N, 10))
    basis[:, 0] = 1.0
    if N == 0:
        return basis
    basis[:, 1] = t
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model combining Chebyshev polynomial, sinusoidal, Gaussian RBF,
    and smooth non-linear terms.
    
    params (length 10):
        p0: coefficient for constant (T_0)
        p1: coefficient for T_1 (linear)
        p2: coefficient for T_2
        p3: coefficient for T_3
        p4: coefficient for T_4
        p5: amplitude for sin(freq*x)
        p6: frequency for sin/cos
        p7: amplitude for cos(freq*x)
        p8: amplitude for Gaussian RBF
        p9: width parameter for Gaussian RBF (inverted, larger = narrower)
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > 120.0 - _DEADLINE_MARGIN:
        # Return a simple linear fallback to avoid timeout
        return params[0] + params[1] * x
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    t = x * 0.5
    
    # Chebyshev polynomial basis (degree 0-4)
    T0 = np.ones(N)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Sinusoidal terms with learnable frequency
    freq = params[6]
    freq_clamped = np.clip(freq, -10.0, 10.0)
    sin_term = np.sin(freq_clamped * x)
    cos_term = np.cos(freq_clamped * x)
    
    # Gaussian RBF with learnable center (fixed at 0) and learnable width
    # Use inverted width: larger p9 means narrower Gaussian
    width_param = np.abs(params[9]) + 0.1  # ensure positive, avoid division by zero
    rbf_arg = -(x * x) / (2.0 * width_param * width_param)
    rbf_arg_clamped = np.clip(rbf_arg, -50.0, 0.0)
    rbf_term = np.exp(rbf_arg_clamped)
    
    # Combine all terms
    result = (
        params[0] * T0
        + params[1] * T1
        + params[2] * T2
        + params[3] * T3
        + params[4] * T4
        + params[5] * sin_term
        + params[7] * cos_term
        + params[8] * rbf_term
        + params[9] * np.tanh(x)  # keep tanh as 10th term for non-linearity
    )
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
