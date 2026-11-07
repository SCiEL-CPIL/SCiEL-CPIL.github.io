import numpy as np

# Define fixed centers for the RBFs
# Domain is [0.5, 2.0]
# Using a mix of grid points and center
centers = np.array([
    [0.75, 0.75, 0.75],
    [1.75, 0.75, 0.75],
    [0.75, 1.75, 0.75],
    [0.75, 0.75, 1.75],
    [1.75, 1.75, 0.75],
    [1.75, 0.75, 1.75],
    [0.75, 1.75, 1.75],
    [1.75, 1.75, 1.75],
    [1.25, 1.25, 1.25],
    [1.25, 0.75, 1.25]
])

# Gamma controls the width of the RBFs
# Larger gamma -> narrower peaks
gamma = 2.0 

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Stack inputs into an array of shape (N, 3)
    points = np.column_stack((x, y, z))
    
    # Compute distances squared from each point to each center
    # points: (N, 3)
    # centers: (10, 3)
    # diff: (N, 10, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    
    # Squared Euclidean distance
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute RBF values: exp(-gamma * dist_sq)
    # Shape: (N, 10)
    rbf_values = np.exp(-gamma * dist_sq)
    
    # Linear combination of RBF values with params
    # params: (10,)
    # rbf_values: (N, 10)
    # result: (N,)
    return rbf_values @ params
