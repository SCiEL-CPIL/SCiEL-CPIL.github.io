import numpy as np

# Normalize inputs to [-1, 1]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

# Define 9 RBF centers on a 3x3 grid in the normalized space
centers = np.array([
    [-1.0, -1.0],
    [-1.0,  0.0],
    [-1.0,  1.0],
    [ 0.0, -1.0],
    [ 0.0,  0.0],
    [ 0.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  0.0],
    [ 1.0,  1.0]
])

# RBF width (sigma)
# A width of 0.5 means the RBF decays significantly over half the domain
sigma = 0.5
gamma = 1.0 / (2.0 * sigma * sigma)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model for the nonlinear oscillator.
    Uses 9 fixed RBF centers on a 3x3 grid in normalized space, 
    plus 1 bias term. Linear in parameters.
    """
    # Normalize inputs
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Stack inputs
    # Shape: (N, 2)
    inputs = np.column_stack((xn, vn))
    
    # Compute distances to centers
    # centers shape: (9, 2)
    # diff shape: (N, 9, 2)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared distances
    sq_dist = np.sum(diff ** 2, axis=2)
    
    # RBF values
    # Shape: (N, 9)
    rbf_vals = np.exp(-gamma * sq_dist)
    
    # Add bias column
    # Shape: (N, 10)
    bias_col = np.ones((len(xn), 1))
    design_matrix = np.hstack([rbf_vals, bias_col])
    
    # Linear combination
    # params shape: (10,)
    result = design_matrix @ params
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
