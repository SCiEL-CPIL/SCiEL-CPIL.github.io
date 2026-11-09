"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Linear combination of Gaussian RBFs centered at distinct points in [-2, 2].
    # This is linear in params, so BFGS converges to the global optimum.
    # 10 centers spread across the domain, with a fixed width chosen for
    # smooth interpolation without overfitting individual points.
    centers = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0])
    width = 0.8  # sigma for Gaussian kernels; ~0.5-1.0 works well on [-2,2]
    
    # Compute Gaussian RBF values: exp(-((x - c)^2) / (2 * width^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Linear combination with params
    # rbf @ params gives shape (n_points,)
    return rbf @ params
# EVOLVE-BLOCK-END
