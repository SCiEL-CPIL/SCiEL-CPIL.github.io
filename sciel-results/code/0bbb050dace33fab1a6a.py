"""
Strategy: Fixed-Width Radial Basis Function (RBF) Model (Linear-in-Params).

Rationale:
1. The current strategy uses a hybrid of low-degree polynomials and low-frequency trigonometric terms. While stable, this global basis may struggle to capture localized features (e.g., sharp peaks, specific non-periodic wiggles) without introducing oscillations elsewhere (Runge-like behavior for polynomials) or being constrained to specific frequencies (trig terms).
2. RBFs use localized basis functions (Gaussians) centered at fixed locations. With 10 centers evenly spaced across [-3, 3], the model can independently adjust the height of the function at each location via the 10 parameters.
3. The model is linear in the parameters: y = sum(w_i * exp(-((x - c_i)^2) / (2 * sigma^2))).
4. Because the centers and width are fixed, the optimization problem is convex quadratic in the weights. BFGS starting from all ones will reliably converge to the global minimum MSE solution for the training data.
5. A global width (sigma) of 1.0 is chosen to provide sufficient overlap between neighboring centers (spacing is 0.667), ensuring the basis is well-conditioned and can represent smooth variations across the entire domain. This avoids the "sparse" problem where isolated Gaussians can't interact to form broad curves, and the "dense" problem where the matrix becomes ill-conditioned.
6. This approach is structurally distinct from polynomial/trig bases, offering a different inductive bias suited for general smooth function approximation on a bounded interval.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed RBF configuration
# 10 centers evenly spaced in [-3, 3]
# Spacing: (3 - (-3)) / (10 - 1) = 6 / 9 = 0.666...
# Centers: -3, -2.333, ..., 3
RBFS_CENTERS = np.linspace(-3.0, 3.0, 10)
# Moderate global width. 
# If sigma is too small, the basis vectors are nearly orthogonal (good for localizing, bad for broad trends).
# If sigma is too large, they become nearly identical (ill-conditioned).
# sigma = 1.0 gives a standard deviation of 1.0. At distance 0.66 (1 spacing), exp(-0.66^2 / 2) ~ 0.8. Good overlap.
RBFS_WIDTH = 1.0
RBFS_INV_2_SIGMA_SQ = 1.0 / (2.0 * RBFS_WIDTH * RBFS_WIDTH)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed-width RBF basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 RBF weights.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure params is a flat array of correct size
    # The evaluator passes a 1D array of length 10.
    if params.size != 10:
        # Safety check, though contract guarantees length 10
        params = np.resize(params, 10)
        
    # Compute the RBF basis matrix
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - RBFS_CENTERS[np.newaxis, :]
    
    # Compute squared distances
    diff_sq = diff * diff
    
    # Compute Gaussian values: exp(-diff_sq * inv_2_sigma_sq)
    # We use exp(-0.5 * (x-c)^2 / sigma^2)
    basis = np.exp(-diff_sq * RBFS_INV_2_SIGMA_SQ)
    
    # Combine basis with weights
    # basis shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
