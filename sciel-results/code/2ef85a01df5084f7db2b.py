"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse linear combination of 10 Radial Basis Functions (RBFs) with fixed centers
    and widths. The optimizer fits only the amplitudes (params[0:10]).
    
    Centers are placed on a deterministic 3x3 grid plus two additional points
    to cover the [-1, 1]^2 domain. Widths are fixed to capture local features.
    """
    # Fixed centers: a 3x3 grid over [-1, 1] gives 9 points, plus 2 more for 10 total
    # Using a Halton-like or structured grid for better coverage
    # 3x3 grid:
    grid_x = np.array([-1.0, 0.0, 1.0])
    grid_y = np.array([-1.0, 0.0, 1.0])
    
    centers = []
    for gx in grid_x:
        for gy in grid_y:
            centers.append([gx, gy])
    # Add two more centers to make 10: e.g., (0.5, 0.5) and (-0.5, -0.5)
    centers.append([0.5, 0.5])
    centers.append([-0.5, -0.5])
    
    centers = np.array(centers)  # shape (10, 2)
    
    # Fixed widths: use a width that captures local structure
    # A moderate width like 0.5 or 0.7 works well for [-1,1] domain
    widths = np.full(10, 0.7)
    
    # Compute RBF values: exp(-width^2 * ||x - center||^2)
    # x shape: (N, 2), centers shape: (10, 2)
    # diff shape: (N, 10, 2)
    diff = x[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)  # shape (N, 10)
    
    # RBF values
    rbf_values = np.exp(-(widths[np.newaxis, :] ** 2) * dist_sq)  # shape (N, 10)
    
    # Amplitudes are params[0:10]
    amplitudes = params[0:10]
    
    # Linear combination
    prediction = rbf_values @ amplitudes
    
    return prediction
# EVOLVE-BLOCK-END
