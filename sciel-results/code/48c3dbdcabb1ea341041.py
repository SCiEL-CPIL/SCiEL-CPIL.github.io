"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate via RBF interpolant with 10 fixed centers."""
    # Normalize inputs to [0,1] based on training ranges
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (39.98 - 1.0)
    temp_norm = (temp - 1.0) / (12.0 - 1.0)
    ph_norm = (pH - 1.0) / (4.999 - 1.0)

    # 10 fixed centers: 4 corners, 4 edge midpoints, 1 centroid, 1 extra
    # Corners of [0,1]^4
    c0 = np.array([0.0, 0.0, 0.0, 0.0])
    c1 = np.array([1.0, 1.0, 1.0, 1.0])
    c2 = np.array([0.0, 1.0, 1.0, 0.0])
    c3 = np.array([1.0, 0.0, 0.0, 1.0])
    # Edge midpoints
    c4 = np.array([0.5, 0.0, 0.0, 0.0])
    c5 = np.array([0.0, 0.5, 0.0, 0.0])
    c6 = np.array([0.0, 0.0, 0.5, 0.0])
    c7 = np.array([0.0, 0.0, 0.0, 0.5])
    # Centroid
    c8 = np.array([0.5, 0.5, 0.5, 0.5])
    # Extra center (high substrate, moderate others)
    c9 = np.array([0.2, 0.8, 0.5, 0.5])

    # Stack centers: shape (10, 4)
    centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9])

    # Stack inputs: shape (N, 4)
    inputs = np.column_stack([b_norm, s_norm, temp_norm, ph_norm])

    # Compute squared distances from each input to each center: shape (N, 10)
    # Using broadcasting: (N,1,4) - (1,10,4) -> (N,10,4)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # shape (N, 10)

    # Gaussian RBF with fixed width sigma=0.5
    sigma = 0.5
    rbf_values = np.exp(-sq_dist / (2.0 * sigma ** 2))  # shape (N, 10)

    # Linear combination: prediction = sum_j w_j * phi_j(x)
    # params shape (10,), rbf_values shape (N, 10)
    growth = rbf_values @ params

    return growth
# EVOLVE-BLOCK-END
