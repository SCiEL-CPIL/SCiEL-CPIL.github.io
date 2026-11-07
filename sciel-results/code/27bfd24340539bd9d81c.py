"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0, 1)."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        i_val = i
        while i_val > 0:
            f /= base
            r += f * (i_val % base)
            i_val //= base
        result.append(r)
    return np.array(result)

def _generate_rbf_centers(n_centers):
    """Generate deterministic RBF centers using Halton sequence over [-1, 1]."""
    x_halton = _halton_sequence(n_centers, 2)
    y_halton = _halton_sequence(n_centers, 3)
    # Map from [0, 1) to [-1, 1]
    x_centers = 2.0 * x_halton - 1.0
    y_centers = 2.0 * y_halton - 1.0
    return np.column_stack([x_centers, y_centers])

# Precompute RBF centers
N_CENTERS = 10
RBF_CENTERS = _generate_rbf_centers(N_CENTERS)
RBF_WIDTH = 0.8  # Fixed width for Gaussian RBFs

def rbf_basis_2d(x, y):
    """
    Returns an array of shape (N, 10) with 10 Gaussian RBF values.
    Each RBF: exp(-||p - c_i||^2 / (2 * width^2))
    """
    N = len(x)
    points = np.column_stack([x, y])
    
    # Compute distances from each point to each center
    # points: (N, 2), RBF_CENTERS: (10, 2)
    # dist_sq: (N, 10)
    diff = points[:, np.newaxis, :] - RBF_CENTERS[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Gaussian RBF: exp(-dist_sq / (2 * width^2))
    basis = np.exp(-dist_sq / (2.0 * RBF_WIDTH ** 2))
    
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if params are valid
    if not np.all(np.isfinite(params)):
        return np.zeros(len(x))
        
    # Compute basis
    basis = rbf_basis_2d(x, y)
    
    # Predict
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(len(x))
        
    return pred
# EVOLVE-BLOCK-END
