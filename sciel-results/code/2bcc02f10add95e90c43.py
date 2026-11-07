"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _halton_sequence(n, base):
    """Generate n points of the Halton sequence in [0, 1) for given base."""
    result = np.zeros(n)
    f = 1.0 / base
    for i in range(1, n + 1):
        x = 0.0
        j = i
        while j > 0:
            j, rem = divmod(j, base)
            x += rem * f
            f /= base
        result[i - 1] = x
    return result


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse linear combination of 9 Radial Basis Functions + 1 constant term.
    
    Centers are fixed by a Halton sequence mapped to [-1, 1]^2.
    The optimizer fits 9 RBF amplitudes and 1 constant (params[9]).
    
    x, y: 1D arrays of length n.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    params = np.asarray(params).ravel()
    
    n = len(x)
    
    # Generate 9 well-distributed centers using 2D Halton sequence
    # Halton in bases 2 and 3 for better distribution
    h1 = _halton_sequence(9, 2)  # x coordinates in [0, 1)
    h2 = _halton_sequence(9, 3)  # y coordinates in [0, 1)
    
    # Map from [0, 1) to [-1, 1]
    centers_x = 2.0 * h1 - 1.0
    centers_y = 2.0 * h2 - 1.0
    
    # Fixed width for all RBFs
    sigma = 0.55
    
    # Compute distances from each point to each center
    dx = x[:, np.newaxis] - centers_x[np.newaxis, :]  # (n, 9)
    dy = y[:, np.newaxis] - centers_y[np.newaxis, :]  # (n, 9)
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))  # (n, 9)
    
    # Prediction: 9 RBF terms + constant
    # params[0:9] are RBF amplitudes, params[9] is constant
    pred = rbf_values @ params[:9] + params[9]
    
    return pred
# EVOLVE-BLOCK-END
