import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 fixed centers for the RBFs, distributed over [-1, 1]^2
    # Using a 3x3 grid plus one center at (0,0) to ensure coverage
    cx = np.array([-1.0, 0.0, 1.0, -0.5, 0.5, -0.5, 0.5, 0.0, -1.0, 1.0])
    cy = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    
    # Precompute squared distances from each input point to each center
    # x and y are arrays of shape (N,)
    # We want a matrix D of shape (N, 10) where D[i, j] = (x_i - cx_j)^2 + (y_i - cy_j)^2
    
    # x[:, np.newaxis] has shape (N, 1)
    # cx[np.newaxis, :] has shape (1, 10)
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # Gaussian RBF: phi(r) = exp(-alpha * r^2)
    # Choosing alpha = 5.0 provides a good balance between local flexibility and smoothness
    # for the domain size. Too small alpha makes it too global (like a single bump), 
    # too large makes it too local (overfitting risk if centers are sparse, but here they are dense).
    alpha = 5.0
    rbf = np.exp(-alpha * dist_sq)
    
    # Linear combination of RBFs with parameters p
    # rbf is (N, 10), params is (10,)
    # Result is (N,)
    return rbf @ params
