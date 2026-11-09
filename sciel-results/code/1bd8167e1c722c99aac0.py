# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of 10 Gaussian basis functions with fixed centers and widths.
    This is a linear model in the amplitudes (params), ensuring BFGS converges
    to the global minimum of the least squares problem.
    Fixed centers and widths avoid the non-convexity of optimizing them,
    which likely caused the poor performance of previous Gaussian attempts.
    """
    # Domain is [0.1, 6]
    # Define 10 centers evenly spaced in the domain
    centers = np.linspace(0.1, 6.0, 10)
    
    # Define a uniform width. 
    # A width of 0.5 provides good overlap between adjacent Gaussians 
    # (distance between centers is ~0.65), ensuring smooth coverage.
    sigma = 0.5
    
    # Compute Gaussian basis functions
    # G[i, j] is the value of the i-th Gaussian at x[j]
    # x shape: (N,)
    # centers shape: (10,)
    # We want G shape: (10, N)
    
    # Expand dimensions for broadcasting
    x_exp = x[:, np.newaxis]  # (N, 1)
    c_exp = centers[np.newaxis, :]  # (1, 10)
    
    # Calculate squared distances
    # diff shape: (N, 10)
    diff = x_exp - c_exp
    
    # Calculate Gaussian values
    # G shape: (N, 10)
    G = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = G @ params
    # G is (N, 10), params is (10,)
    # Result y is (N,)
    y = G @ params
    
    return y
# EVOLVE-BLOCK-END
