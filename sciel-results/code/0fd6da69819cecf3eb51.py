"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed hyperparameters for the RBF basis
# Domain is [0.1, 6]
# We use 8 RBF centers evenly spaced
_CENTERS = np.linspace(0.1, 6.0, 8)
# Fixed bandwidth
_SIGMA = 0.5
# Pre-compute the denominator for the exponent to save time
_INV_2_SIGMA2 = 1.0 / (2.0 * _SIGMA * _SIGMA)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the model prediction.
    
    Model: y = p[0] + p[1]*x + sum(p[2+i] * exp(-(x-c_i)^2 / (2*sigma^2)))
    
    Args:
    x: Input array of shape (N,)
    params: Parameter array of shape (10,)
        - params[0]: Constant offset
        - params[1]: Linear coefficient
        - params[2:10]: RBF weights
    
    Returns:
    Prediction array of shape (N,)
    """
    # Ensure x is a 1D array for calculation, though it might come as column
    x_flat = np.asarray(x).flatten()
    
    # params mapping
    w_const = params[0]
    w_linear = params[1]
    w_rbf = params[2:10]
    
    # Compute RBF values
    # x_flat shape: (N,)
    # _CENTERS shape: (8,)
    # diff shape: (N, 8)
    diff = x_flat[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Exponent part
    # exp_arg shape: (N, 8)
    exp_arg = -diff * diff * _INV_2_SIGMA2
    
    # RBF matrix shape: (N, 8)
    rbf_matrix = np.exp(exp_arg)
    
    # Linear combination
    # rbf_matrix @ w_rbf -> shape (N,)
    y_rbf = rbf_matrix @ w_rbf
    
    # Total prediction
    y = w_const + w_linear * x_flat + y_rbf
    
    return y
# EVOLVE-BLOCK-END
