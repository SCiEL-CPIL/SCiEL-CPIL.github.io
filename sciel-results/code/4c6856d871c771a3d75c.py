import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed set of Gaussian Radial Basis Functions (RBFs).
    
    This approach uses 10 fixed centers in the input space [-1, 1]^2.
    The model is linear in the parameters (weights), ensuring the optimization
    landscape is convex, which helps BFGS converge reliably.
    RBFs can capture local features and non-polynomial structures that global
    polynomial bases might miss or approximate poorly with only 10 terms.
    
    Centers are chosen to cover the domain:
    5 x-coordinates: -1.0, -0.5, 0.0, 0.5, 1.0
    2 y-coordinates: -1.0, 1.0
    Total 10 centers.
    
    Width (sigma) is fixed to 0.5 to provide overlap between basis functions
    while maintaining locality.
    """
    # Fixed centers
    centers_x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    centers_y = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Compute distances from each input point to each center
    # x and y are arrays of shape (N,)
    # centers_x and centers_y are arrays of shape (10,)
    
    # dx shape: (N, 10)
    dx = x[:, np.newaxis] - centers_x[np.newaxis, :]
    dy = y[:, np.newaxis] - centers_y[np.newaxis, :]
    
    # Squared distances
    dist_sq = dx**2 + dy**2
    
    # Gaussian RBFs with sigma = 0.5
    # phi = exp(-dist_sq / (2 * sigma^2))
    sigma_sq_2 = 2 * (0.5**2) # 0.5
    basis = np.exp(-dist_sq / sigma_sq_2)
    
    # Linear combination
    # basis shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    f = basis @ params
    
    return f
