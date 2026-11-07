# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation: Sum of 10 Gaussian basis functions with FIXED centers and widths.
    
    f(x,y,z) = sum_{i=0}^{9} w_i * exp(-((x-c_xi)^2 + (y-c_yi)^2 + (z-c_zi)^2) / (2 * sigma^2))
    
    Parameters (10 total):
    - params[0:10] : Weights w_0 ... w_9
    
    This model is LINEAR in parameters. BFGS will converge to the global optimum
    for this basis. The fixed centers are chosen to cover the domain [0.5, 2.0]^3.
    """
    # Fixed centers covering the domain corners and some interior points
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 0.5, 1.25],
        [1.25, 2.0, 1.25]
    ])
    
    # Fixed width
    sigma = 0.7
    sigma2 = sigma * sigma
    
    # Compute basis functions
    # x, y, z are arrays of shape (N,)
    # centers is (10, 3)
    # We need dist_sq for each point and each center
    
    # Efficient computation using broadcasting
    # x[:, None] - centers[None, :, 0]
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    dz = z[:, None] - centers[None, :, 2]
    
    dist_sq = dx**2 + dy**2 + dz**2
    
    # Basis values: (N, 10)
    basis = np.exp(-dist_sq / (2.0 * sigma2))
    
    # Linear combination: (N, 10) dot (10,) -> (N,)
    return basis @ params
# EVOLVE-BLOCK-END
