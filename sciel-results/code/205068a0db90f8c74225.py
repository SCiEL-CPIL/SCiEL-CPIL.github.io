"""Symbolic regression with a rich, well-conditioned basis and safe evaluation."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed RBF centers and widths
# Centers spread uniformly from -3 to 3
_CENTERS = np.linspace(-3.0, 3.0, 10)
# Widths: mix of narrow and wide to capture both local and global trends
# Using a geometric progression for widths
_WIDTHS = np.linspace(0.5, 3.0, 10)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a 10-term RBF basis.
    Each term is an RBF: exp(-((x - c_i)^2) / (2 * sigma_i^2))
    params are the weights for these 10 RBFs.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()

    # Guard against non-finite inputs
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)

    p = np.asarray(params, dtype=float)
    if p.shape != (10,):
        p = np.resize(p, 10)
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Compute RBF basis
    # x is (N,), _CENTERS is (10,)
    # We want result (N, 10)
    # Using broadcasting: x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    # Squared distances
    dist_sq = diff * diff
    # Normalize by 2 * sigma^2
    # _WIDTHS is (10,), so _WIDTHS[np.newaxis, :] is (1, 10)
    sigma_sq = _WIDTHS[np.newaxis, :] ** 2
    # Exponent
    exponent = -dist_sq / (2.0 * sigma_sq)
    # Basis matrix (N, 10)
    basis = np.exp(exponent)
    
    # Weighted sum: (N, 10) dot (10,) -> (N,)
    result = basis @ p

    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
