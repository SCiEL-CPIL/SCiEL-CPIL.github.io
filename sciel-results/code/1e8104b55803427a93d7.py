import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Extract params
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    w1, mu1, sigma1_raw = params[4], params[5], params[6]
    w2, mu2, sigma2_raw = params[7], params[8], params[9]
    
    # Safe sigma
    sigma1 = np.exp(sigma1_raw)
    sigma2 = np.exp(sigma2_raw)
    
    # Chebyshev part
    t = x / 2.0
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    
    poly_part = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    
    # Gaussian part
    gauss1 = w1 * np.exp(-(x - mu1)**2 / (2.0 * sigma1**2))
    gauss2 = w2 * np.exp(-(x - mu2)**2 / (2.0 * sigma2**2))
    
    return poly_part + gauss1 + gauss2
