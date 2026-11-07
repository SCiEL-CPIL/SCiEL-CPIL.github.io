import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Polynomial basis + Gaussian RBFs.
    params[0:6] are coefficients for [1, x, v, x^2, v^2, xv]
    params[6:10] are weights for 4 Gaussian RBFs.
    """
    # Polynomial terms
    p0, p1, p2, p3, p4, p5 = params[0], params[1], params[2], params[3], params[4], params[5]
    
    poly = p0 + p1 * x + p2 * v + p3 * x**2 + p4 * v**2 + p5 * x * v
    
    # RBF terms
    # Fixed centers
    c1 = np.array([-0.3, -0.1])
    c2 = np.array([0.3, -0.1])
    c3 = np.array([-0.3, 0.1])
    c4 = np.array([0.3, 0.1])
    
    # Fixed width
    sigma = 0.3
    
    # Calculate squared distances
    # d^2 = (x - cx)^2 + (v - cv)^2
    d1_sq = (x - c1[0])**2 + (v - c1[1])**2
    d2_sq = (x - c2[0])**2 + (v - c2[1])**2
    d3_sq = (x - c3[0])**2 + (v - c3[1])**2
    d4_sq = (x - c4[0])**2 + (v - c4[1])**2
    
    # Gaussian kernels
    k1 = np.exp(-d1_sq / (2 * sigma**2))
    k2 = np.exp(-d2_sq / (2 * sigma**2))
    k3 = np.exp(-d3_sq / (2 * sigma**2))
    k4 = np.exp(-d4_sq / (2 * sigma**2))
    
    # Combine
    rbf_part = params[6] * k1 + params[7] * k2 + params[8] * k3 + params[9] * k4
    
    return poly + rbf_part
