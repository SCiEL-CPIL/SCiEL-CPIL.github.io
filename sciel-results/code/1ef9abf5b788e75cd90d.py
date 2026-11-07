import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a sum of Gaussian bumps in state space
    and a Fourier series in time.
    
    Structure:
    a = sum_{i=1}^{5} w_i * exp(-((X-cx_i)^2 + (V-cv_i)^2) / (2*sigma^2)) 
        + c_0 + c_1*cos(t) + c_2*sin(t) + c_3*cos(2t) + c_4*sin(2t)
    
    Inputs are scaled to O(1).
    """
    # Scale inputs to improve conditioning
    # t range [30, 50]
    # x range [-0.2056, 0.1749] -> scale by 0.2 -> X in approx [-1, 1]
    # v range [-0.3391, 0.3624] -> scale by 0.35 -> V in approx [-1, 1]
    
    X = x / 0.2
    V = v / 0.35
    
    # Define 5 fixed Gaussian centers in the scaled (X, V) space
    # Centers chosen to cover the operating range around the origin
    centers = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [-0.5, 0.0],
        [0.0, 0.5],
        [0.0, -0.5]
    ])
    
    sigma = 0.5
    
    # Compute Gaussian basis functions
    # Shape: (N, 5)
    # X, V are (N,)
    # centers are (5, 2)
    # Correctly compute distance from each point (X_i, V_i) to each center (cx_j, cv_j)
    dx = X[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = V[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    # dist_sq is (N, 5)
    dist_sq = dx**2 + dy**2
    gaussians = np.exp(-dist_sq / (2 * sigma**2))
    
    # Time basis functions
    # 1, cos(t), sin(t), cos(2t), sin(2t)
    time_basis = np.column_stack((
        np.ones_like(t),
        np.cos(t),
        np.sin(t),
        np.cos(2 * t),
        np.sin(2 * t)
    ))
    
    # Combine basis functions
    # Total basis: (N, 10)
    basis = np.hstack((gaussians, time_basis))
    
    # Linear combination
    pred = basis @ params
    
    return pred

# EVOLVE-BLOCK-END
