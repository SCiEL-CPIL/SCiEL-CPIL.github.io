"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()

# Fixed RBF centers in the native x-domain [-3, 3].
# Chosen to give good spatial coverage without overlap-induced ill-conditioning.
_RBF_CENTERS = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
_RBF_WIDTH = 0.8  # fixed Gaussian width; tuned for smooth local bumps on [-3,3]


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smooth trend) + fixed-center
    Gaussian RBFs (local features).

    params[0..9]  : 10 Chebyshev coefficients T_0..T_9 (well-conditioned global fit)
    params[10..15]: 6 Gaussian RBF amplitudes at fixed centers (local detail)

    The Chebyshev part keeps the problem close to a well-behaved linear/quadratic
    surface so BFGS converges reliably; the RBFs add non-polynomial flexibility
    to capture sharp or non-smooth features that a degree-9 polynomial cannot.
    All basis functions are fixed (no learnable centers/widths), so the model is
    exactly linear in params and BFGS sees a smooth, bounded objective.
    """
    # Deadline guard: stay safely inside the 150 s wall-clock budget.
    if time.time() - _T0 > 145.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

    # ---- Chebyshev part (global trend) ----
    # Map x from [-3, 3] to t in [-1, 1] for the Chebyshev domain.
    t = x / 3.0
    t = np.clip(t, -1.0, 1.0)
    t = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)

    # Compute Chebyshev polynomials T_0 ... T_9 via the recurrence
    # T_0 = 1, T_1 = t, T_{k+1} = 2*t*T_k - T_{k-1}
    T_prev = np.ones_like(t)   # T_0
    T_curr = t.copy()          # T_1
    result = params[0] * T_prev
    if len(params) > 1:
        result = result + params[1] * T_curr

    for k in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
        result = result + params[k] * T_curr

    # ---- Gaussian RBF part (local detail) ----
    # Each RBF: exp(-((x - c) / width)^2); fixed centers and width.
    # Precompute the squared distances in one vectorized pass.
    diff = x[:, None] - _RBF_CENTERS[None, :]
    r2 = diff * diff
    rbf_matrix = np.exp(-r2 / (_RBF_WIDTH * _RBF_WIDTH))

    # Amplitudes are params[10..15]; use zero if fewer params supplied.
    if len(params) > 10:
        amps = params[10:16]
        # Guard against a short params array (shouldn't happen; evaluator gives 16).
        if len(amps) < 6:
            padded = np.zeros(6, dtype=np.float64)
            padded[:len(amps)] = amps
            amps = padded
        rbf_contrib = rbf_matrix @ amps
    else:
        rbf_contrib = 0.0

    result = result + rbf_contrib

    # Ensure finite output for robustness.
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    return result
# EVOLVE-BLOCK-END
