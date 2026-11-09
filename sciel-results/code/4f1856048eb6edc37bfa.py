import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    
    # RBF Basis: 1 constant + 9 Gaussians
    # Centers in a 3x3 grid on [-1, 1] x [-1, 1]
    centers = np.array([
        [-0.8, -0.8],
        [-0.8,  0.0],
        [-0.8,  0.8],
        [ 0.0, -0.8],
        [ 0.0,  0.0],
        [ 0.0,  0.8],
        [ 0.8, -0.8],
        [ 0.8,  0.0],
        [ 0.8,  0.8]
    ])
    
    # Width parameter for the Gaussians
    sigma = 0.5
    sigma2 = 2 * sigma**2
    
    # Reshape for broadcasting
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    
    cx = centers[:, 0][np.newaxis, :]
    cy = centers[:, 1][np.newaxis, :]
    
    # Squared Euclidean distance
    dist_sq = (x_col - cx)**2 + (y_col - cy)**2
    
    # Gaussian values
    gaussians = np.exp(-dist_sq / sigma2)
    
    # params[0] is the constant term
    # params[1:10] are the weights for the 9 Gaussians
    constant = params[0]
    weights = params[1:]
    
    pred = constant + gaussians @ weights
    
    return pred
