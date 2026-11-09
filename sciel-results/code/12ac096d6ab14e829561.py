"""
Strategy: Fixed-Center RBF Interpolant.
Instead of letting BFGS optimize the centers of the Gaussian RBFs (which is a
non-convex, multi-modal optimization problem prone to local minima and slow
convergence), we fix the centers at evenly spaced points across the input domain
[-3, 3]. We use 5 centers. The 10 parameters are split into:
- 5 weights (params[0:5])
- 5 widths (params[5:10]), transformed to be positive.

This reduces the optimization to a convex-like problem (linear in weights,
smooth in widths) that BFGS can solve very quickly and reliably from the
initialization point (all ones). This avoids the "120s timeout" risk associated
with difficult non-linear fits and ensures a valid, smooth prediction.
"""
import numpy as np

# Pre-compute fixed centers
# Domain is [-3, 3]. 5 centers evenly spaced.
# -3, -1.5, 0, 1.5, 3
_FIXED_CENTERS = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-Center RBF model.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            - params[0:5]: Weights for the 5 Gaussians.
            - params[5:10]: Raw widths for the 5 Gaussians.
    
    Returns:
        Prediction array of shape (N,).
    """
    # Extract weights and raw widths
    weights = params[0:5]
    raw_widths = params[5:10]
    
    # Transform raw widths to positive values.
    # Using exp ensures positivity. Clipping prevents overflow/underflow issues
    # during the optimization steps if params go far from 1.
    # exp(1) ~= 2.7, which is a reasonable initial width for domain size 6.
    widths = np.exp(np.clip(raw_widths, -5.0, 5.0))
    
    # Compute RBF values
    # x shape: (N,)
    # _FIXED_CENTERS shape: (5,)
    # We want a matrix of shape (N, 5) where element (i, j) is the RBF value at x[i] with center c[j].
    
    # Efficient vectorized computation
    # diffs shape: (N, 5)
    diffs = x[:, np.newaxis] - _FIXED_CENTERS[np.newaxis, :]
    
    # rbf_values shape: (N, 5)
    # g(x, c, w) = exp(-(x-c)^2 / (2 * w^2))
    # Note: w is std dev.
    # We need to broadcast widths correctly.
    # widths shape: (5,) -> (1, 5) for broadcasting against (N, 5)
    inv_2_w_sq = 1.0 / (2.0 * widths[np.newaxis, :] ** 2)
    
    rbf_values = np.exp(-diffs ** 2 * inv_2_w_sq)
    
    # Combine: y = sum_j (weights[j] * rbf_values[:, j])
    # rbf_values shape: (N, 5)
    # weights shape: (5,)
    predictions = rbf_values @ weights
    
    return predictions
