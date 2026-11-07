import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params is expected to be of length 10
    # New Basis: 1, x, y, z, 1/x, 1/y, 1/z, xy, xz, yz
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = z
    term4 = 1.0 / x
    term5 = 1.0 / y
    term6 = 1.0 / z
    term7 = x * y
    term8 = x * z
    term9 = y * z
    
    # Combine
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + 
            p7 * term7 + p8 * term8 + p9 * term9)
            
    return pred
