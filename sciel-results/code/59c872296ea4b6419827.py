import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0:4] = RBF weights at 4 corner centers
    # params[4:10] = polynomial coefficients: 1, x, y, x^2, y^2, xy
    
    # RBF centers at the four corners of [-1,1]^2
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0]
    ])
    
    sigma = 0.8
    
    # Compute RBF contributions
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]  # n x 4
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]  # n x 4
    dist_sq = dx**2 + dy**2  # n x 4
    
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))  # n x 4
    
    # Polynomial basis
    poly = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x**2,
        y**2,
        x * y
    ])  # n x 6
    
    # Combine: RBF part + polynomial part
    rbf_part = rbf_values @ params[0:4]       # n
    poly_part = poly @ params[4:10]            # n
    
    return rbf_part + poly_part
