import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Polynomial part
    # p0..p4
    p_poly = params[:5]
    p_rbf = params[5:]
    
    # Calculate polynomial
    y = np.zeros_like(x)
    y += p_poly[0] * np.ones_like(x)
    y += p_poly[1] * x
    y += p_poly[2] * x**2
    y += p_poly[3] * x**3
    y += p_poly[4] * x**4
    
    # RBF part
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    # Vectorized calculation of Gaussians
    # x is (N,), centers is (5,)
    # We want (N, 5) matrix of distances
    # (x[:, np.newaxis] - centers[np.newaxis, :])**2
    dist_sq = (x[:, np.newaxis] - centers[np.newaxis, :])**2
    gaussians = np.exp(-dist_sq / (2 * sigma**2))
    
    y += gaussians @ p_rbf
    
    return y
