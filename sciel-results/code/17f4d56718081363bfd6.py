"""Symbolic regression with 3 inputs - Gaussian RBF basis approach."""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a linear combination of 10 Gaussian basis functions.
    The centers are fixed to cover the input domain [0.5, 2]^3.
    The width is fixed to sigma = 0.6.
    params: array of 10 weights.
    """
    p = params.flatten()
    
    # Define 10 fixed centers in the domain [0.5, 2]
    # We want good coverage.
    # Let's use a grid-like distribution.
    # 2 points per dimension would be 8 points. We have 10.
    # Let's add the center (1,1,1) and one more.
    
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([1.5, 0.5, 0.5])
    c3 = np.array([0.5, 1.5, 0.5])
    c4 = np.array([0.5, 0.5, 1.5])
    c5 = np.array([1.5, 1.5, 0.5])
    c6 = np.array([1.5, 0.5, 1.5])
    c7 = np.array([0.5, 1.5, 1.5])
    c8 = np.array([1.5, 1.5, 1.5])
    c9 = np.array([1.0, 1.0, 1.0])
    c10 = np.array([1.0, 1.0, 0.5]) # Extra point to break symmetry
    
    centers = np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # Inputs are (x, y, z) arrays. Stack them.
    # x, y, z are 1D arrays of length N.
    inputs = np.stack([x, y, z], axis=1) # Shape (N, 3)
    
    # Calculate distances from each input point to each center
    # dist_sq[i, j] = ||inputs[i] - centers[j]||^2
    # Use broadcasting: (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
    diffs = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2) # Shape (N, 10)
    
    # Gaussian kernel
    sigma = 0.6
    basis = np.exp(-dist_sq / (2 * sigma ** 2)) # Shape (N, 10)
    
    # Linear combination
    # result[i] = sum_j basis[i, j] * p[j]
    result = basis @ p
    
    return result
