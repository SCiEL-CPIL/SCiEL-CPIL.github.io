import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to approximately [-1, 1] to improve numerical conditioning
    # Estimated ranges: x ~ [-0.6, 0.6], v ~ [-0.3, 0.3]
    X = x / 0.6
    V = v / 0.3
    
    # Compute powers of scaled variables
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Combine terms with fitted parameters
    # The structure remains a full polynomial up to degree 3, but on scaled inputs
    return p0 + p1 * X + p2 * V + p3 * X2 + p4 * X3 + p5 * V2 + p6 * V3 + p7 * XV + p8 * X2V + p9 * XV2
