import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 3 Gaussian kernels and a bias term.
    params: [b, w1, cx1, cy1, w2, cx2, cy2, w3, cx3, cy3]
    """
    x = x.flatten()
    y = y.flatten()
    
    b = params[0]
    gamma = 1.0
    
    # Initialize prediction with bias
    pred = np.full_like(x, b, dtype=np.float64)
    
    # RBF 1
    w1, cx1, cy1 = params[1], params[2], params[3]
    dist1_sq = (x - cx1)**2 + (y - cy1)**2
    pred += w1 * np.exp(-gamma * dist1_sq)
    
    # RBF 2
    w2, cx2, cy2 = params[4], params[5], params[6]
    dist2_sq = (x - cx2)**2 + (y - cy2)**2
    pred += w2 * np.exp(-gamma * dist2_sq)
    
    # RBF 3
    w3, cx3, cy3 = params[7], params[8], params[9]
    dist3_sq = (x - cx3)**2 + (y - cy3)**2
    pred += w3 * np.exp(-gamma * dist3_sq)
    
    return pred
