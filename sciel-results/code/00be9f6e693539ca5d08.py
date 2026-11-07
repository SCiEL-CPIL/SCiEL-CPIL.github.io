"""
Structural change: single hidden layer RBF-MLP with fixed Gaussian activations
at 10 evenly-spaced centers, 10 parameters as output weights (no separate bias).
Linear in params -> BFGS converges to global optimum for this basis.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF hidden-layer model: 10 Gaussian basis functions, 10 output weights.
    Each parameter is the weight of one RBF centered at evenly-spaced points
    in [0.1, 6]. No separate constant term (the constant is captured by the
    combination of RBFs).
    
    params: array of length 10, each is the weight for one RBF.
    """
    # 10 centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    # Fixed width - slightly wider to allow more global coverage
    width = 1.5
    
    # Build RBF matrix: shape (n_points, 10)
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 10)
    
    # Compute squared distances
    dist_sq = (x_expanded - centers_expanded) ** 2  # (n, 10)
    
    # RBF values: exp(-dist_sq / (2 * width^2))
    rbf_values = np.exp(-dist_sq / (2.0 * width ** 2))  # (n, 10)
    
    # Output: weighted sum of all 10 RBFs
    prediction = rbf_values @ params  # (n,)
    
    return prediction
