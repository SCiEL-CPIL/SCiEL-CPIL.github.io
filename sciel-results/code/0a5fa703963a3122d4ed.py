import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense, fixed Radial Basis Function (RBF) basis.
    
    Strategy:
    - Use 10 Gaussian RBFs centered at equally spaced points in [-2, 2].
    - This provides a flexible, local basis that can approximate a wide variety
      of continuous functions, unlike the global polynomial/trig basis.
    - The width of the Gaussians is fixed to ensure the basis is well-spread
      across the domain, allowing the linear combination to capture local features.
    - BFGS will optimize the 10 weights to fit the target function.
    """
    # Define 10 centers in [-2, 2]
    # We use 10 centers to match the 10 parameters.
    # Centers are spaced evenly: -2, -1.55, -1.1, ..., 2
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for the Gaussian RBFs.
    # A width of 0.5 provides good local coverage without being too narrow
    # (which would make the problem ill-conditioned) or too wide (which would
    # make it behave like a single global bump).
    sigma = 0.5
    
    # Compute the RBF basis values: phi_j(x) = exp(-((x - c_j)^2) / (2 * sigma^2))
    # x is shape (N,), centers is shape (10,)
    # We want a matrix of shape (N, 10) where each row is the RBF values for one x_i
    # and each column corresponds to one center c_j.
    
    # Compute squared distances
    # (x_i - c_j)^2 for all i, j
    # x is (N, 1), centers is (10,) -> broadcast to (N, 10)
    x_col = x.reshape(-1, 1)
    dist_sq = (x_col - centers.reshape(1, -1)) ** 2
    
    # Compute RBF values
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # rbf_values is shape (N, 10)
    # params is shape (10,)
    # We compute the weighted sum: prediction_i = sum_j params[j] * rbf_values[i, j]
    # This is a matrix-vector product: (N, 10) @ (10,) -> (N,)
    predictions = rbf_values @ params
    
    return predictions
