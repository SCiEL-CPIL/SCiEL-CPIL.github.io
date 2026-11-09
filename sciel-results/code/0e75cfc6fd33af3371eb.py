import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF model with Gaussian kernels + Polynomial trend
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    # 5 params for RBFs, 5 params for polynomial trend (1, x, y, x^2, y^2)
    
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Calculate squared distances from each center
    # x and y are vectors of length N
    # cx and cy are vectors of length 5
    # We need distance from each point (xi, yi) to each center (cxj, cyj)
    
    # Efficient vectorization:
    # dx = x[:, None] - cx[None, :]  # Shape (N, 5)
    # dy = y[:, None] - cy[None, :]  # Shape (N, 5)
    # dist_sq = dx**2 + dy**2        # Shape (N, 5)
    
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernel with sigma = 1.0
    # K = exp(-dist_sq / (2 * sigma^2))
    sigma = 1.0
    K = np.exp(-dist_sq / (2.0 * sigma**2))  # Shape (N, 5)
    
    # RBF part: params[0:5] dot K
    rbf_pred = K @ params[0:5]  # Shape (N,)
    
    # Polynomial trend part: params[5:10]
    # p5: 1, p6: x, p7: y, p8: x^2, p9: y^2
    poly_pred = (params[5] * np.ones_like(x) + 
                 params[6] * x + 
                 params[7] * y + 
                 params[8] * (x**2) + 
                 params[9] * (y**2))
                 
    return rbf_pred + poly_pred
