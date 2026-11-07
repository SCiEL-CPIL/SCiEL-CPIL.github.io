"""Multi-scale Gaussian RBF network: 10 params as output weights over 10 centers x 5 widths."""
# EVOLVE-BLOCK-START
import numpy as np


# Precompute fixed RBF basis (independent of x and params) for speed.
# 10 centers evenly spaced in [0.1, 6].
_CENTERS = np.linspace(0.1, 6.0, 10)          # shape (10,)
# 5 fixed widths: local to global.
_WIDTHS = np.array([0.2, 0.4, 0.8, 1.5, 2.5])  # shape (5,)
# Centering constant to keep arguments well-scaled.
_X_CENTER = 3.05


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale RBF model.
    params (length 10): output weights, one per center.
    For each center c_i, we sum Gaussian RBFs at 5 fixed widths, then
    multiply the sum by params[i]. This lets a single weight per center
    control a blend of local and global bumps, giving a rich, smooth,
    non-linear function class that is linear (hence well-conditioned)
    in the parameters for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Center x for numerical conditioning.
    xc = x - _X_CENTER                      # shape (n,)

    # Build RBF basis: for each point and each center, sum over widths.
    # dist_sq: (n, 10)  = (xc[:,None] - centers[None,:])^2
    dist_sq = (xc[:, np.newaxis] - _CENTERS[np.newaxis, :]) ** 2

    # For each width w, compute exp(-dist_sq / (2 w^2)) -> (n, 10)
    # Stack over widths: (5, n, 10)
    # Use broadcasting: widths (5,1,1), dist_sq (1,n,10)
    inv_2w2 = 1.0 / (2.0 * _WIDTHS[:, np.newaxis, np.newaxis] ** 2)  # (5,1,1)
    rbf_stack = np.exp(-inv_2w2 * dist_sq[np.newaxis, :, :])        # (5, n, 10)

    # Sum over widths -> basis (n, 10): each column is the multi-scale RBF for one center.
    basis = rbf_stack.sum(axis=0)          # (n, 10)

    # Linear combination with params.
    return basis @ params
# EVOLVE-BLOCK-END
