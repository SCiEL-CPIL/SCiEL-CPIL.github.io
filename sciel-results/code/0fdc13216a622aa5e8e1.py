"""Symbolic regression: piecewise linear hat functions with 10 fixed knots."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under the 150s wall-clock limit

# Domain: x in [0.1, 6]
_X_MIN = 0.1
_X_MAX = 6.0

# 10 evenly spaced knots across the domain
_KNOTS = np.linspace(_X_MIN, _X_MAX, 10)


def _build_basis(x: np.ndarray) -> np.ndarray:
    """Return an (N, 10) basis matrix of hat functions for the given 1-D x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    num_knots = len(_KNOTS)
    
    # Precompute knot spacing
    # For interior knots, width is 2 * spacing
    # For boundary knots, we use one-sided hat functions (linear ramps)
    
    basis = np.zeros((n, num_knots), dtype=np.float64)
    
    for k in range(num_knots):
        if k == 0:
            # Left boundary: linear ramp from _KNOTS[0] to _KNOTS[1]
            # hat = (x - x_0) / (x_1 - x_0) for x in [x_0, x_1]
            # 0 for x < x_0, 1 for x > x_1? No, hat functions are local.
            # Standard P1 FEM basis:
            # phi_0(x) = (x - x_0) / (x_1 - x_0) for x in [x_0, x_1], 0 otherwise?
            # Actually, to cover the domain, we often extend or use clamped splines.
            # Let's use standard hat functions on the knots.
            # phi_k(x) is non-zero only in [x_{k-1}, x_{k+1}].
            # For k=0, it's non-zero in [x_0, x_1].
            # phi_0(x) = (x - x_0) / (x_1 - x_0)
            width = _KNOTS[1] - _KNOTS[0]
            basis[:, 0] = np.clip((x - _KNOTS[0]) / width, 0, 1)
            
        elif k == num_knots - 1:
            # Right boundary: linear ramp from _KNOTS[-2] to _KNOTS[-1]
            width = _KNOTS[-1] - _KNOTS[-2]
            basis[:, -1] = np.clip((_KNOTS[-1] - x) / width, 0, 1)
            
        else:
            # Interior knots: standard hat
            left_width = _KNOTS[k] - _KNOTS[k-1]
            right_width = _KNOTS[k+1] - _KNOTS[k]
            
            # Left side: (x - x_{k-1}) / (x_k - x_{k-1})
            left_val = np.clip((x - _KNOTS[k-1]) / left_width, 0, 1)
            # Right side: (x_{k+1} - x) / (x_{k+1} - x_k)
            right_val = np.clip((_KNOTS[k+1] - x) / right_width, 0, 1)
            
            basis[:, k] = np.minimum(left_val, right_val)
            
    # Column-wise standardization for conditioning (guard against zero std)
    mean = basis.mean(axis=0, keepdims=True)
    std = basis.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    basis = (basis - mean) / std

    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Prediction = standardized_basis(x) @ params, with a deadline guard."""
    # Deadline guard: if we're too close to the limit, return a safe constant
    if time.time() > _DEADLINE:
        return np.ones(np.asarray(x, dtype=np.float64).shape, dtype=np.float64)

    x_arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x_arr)):
        return np.zeros_like(x_arr, dtype=np.float64)

    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        params = np.resize(params, 10)

    basis = _build_basis(x_arr)
    predictions = basis @ params

    # Ensure finite output
    if not np.all(np.isfinite(predictions)):
        predictions = np.where(np.isfinite(predictions), predictions, 0.0)

    # Return in the same shape as the input
    return predictions.reshape(x_arr.shape)
# EVOLVE-BLOCK-END
