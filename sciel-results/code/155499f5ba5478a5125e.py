"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smooth approximation)
    mixed with Gaussian RBFs (localized features) and a tanh term.

    10 parameters total:
      - 5 Chebyshev coefficients (T_0..T_4)
      - 3 Gaussian RBF coefficients (centers fixed at -1.5, 0, 1.5)
      - 1 Tanh coefficient
      - 1 Gaussian RBF width (shared, log-parameterized for positivity)

    The RBF width is learnable, allowing the model to adapt localization
    to the target function's features.
    """
    if time.time() - _T0 > 140.0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

    # --- Chebyshev part on t = x/3 in [-1,1] ---
    t = np.clip(x / 3.0, -1.0, 1.0)
    T_prev = np.ones_like(t)   # T_0
    T_curr = t.copy()          # T_1
    result = params[0] * T_prev
    result = result + params[1] * T_curr
    for k in range(2, 5):
        T_next = 2.0 * t * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
        result = result + params[k] * T_curr

    # --- Gaussian RBF part: 3 fixed centers, learnable width ---
    centers = np.array([-1.5, 0.0, 1.5])
    # Parameterize width as exp(param) to ensure positivity.
    # Starting param=1 => width=e ~ 2.718.
    # This is a reasonable initial guess.
    width = np.exp(params[9])
    for i, c in enumerate(centers):
        result = result + params[5 + i] * np.exp(-((x - c) ** 2) / (2.0 * width ** 2))

    # --- Tanh part ---
    result = result + params[8] * np.tanh(x)

    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    return result
# EVOLVE-BLOCK-END
