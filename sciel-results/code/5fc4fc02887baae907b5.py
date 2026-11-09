import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Radial Basis Function (RBF) model with Gaussian kernels
    # Centers at four corners of [-1,1]^2 and the origin
    # 10 parameters: 5 for Gaussian weights, 5 for polynomial terms
    
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Tighter Gaussian width for more local flexibility
    # Observation: 0.8 -> 7.0, 0.6 -> 2.4. Trying 0.5.
    sigma = 0.5
    
    # Compute Gaussian kernels for each center
    # For efficiency, compute squared distances
    b_gauss = np.zeros_like(x)
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx**2 + dy**2
        b_gauss += params[i] * np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Polynomial terms: Constant, linear, and quadratic
    # Replacing xy with constant 1 to allow for global offset
    b_lin = params[5] * 1.0 + params[6] * x + params[7] * y + params[8] * x**2 + params[9] * y**2
    
    pred = b_gauss + b_lin
    
    return pred
