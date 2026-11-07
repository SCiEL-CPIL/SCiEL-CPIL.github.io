import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Gaussian Radial Basis Functions.
    10 fixed RBF centers on a grid in scaled (x,v) space, with shared width.
    Only the 10 linear combination weights are fitted by BFGS.
    This is linear-in-parameters, making optimization well-conditioned.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1]
    X = (x + 0.02265) / 0.55015
    V = (v - 0.05335) / 0.22875
    
    # 10 fixed RBF centers on a grid
    centers_x = np.array([-1.0, -1.0, -0.33, 0.33, 1.0,
                          -1.0, -0.33, 0.33, 1.0, 0.0])
    centers_v = np.array([-1.0, 0.0, 1.0, -1.0, 0.0,
                           1.0, 0.0, -1.0, 0.0, 0.0])
    
    # Shared width for all RBFs
    sigma = 0.8
    
    # Compute RBF values: exp(-((X-cx)^2 + (V-cv)^2) / (2*sigma^2))
    dx = X[:, np.newaxis] - centers_x[np.newaxis, :]
    dv = V[:, np.newaxis] - centers_v[np.newaxis, :]
    dist_sq = dx**2 + dv**2
    
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Linear combination
    return rbf_values @ params
