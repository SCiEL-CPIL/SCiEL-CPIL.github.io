import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] for better conditioning
    t = x / 3.0
    
    # Basis 1: Polynomial terms (degree 0 to 4)
    # p0 = 1
    # p1 = t
    # p2 = t^2
    # p3 = t^3
    # p4 = t^4
    P0 = np.ones_like(t)
    P1 = t
    P2 = t * t
    P3 = P2 * t
    P4 = P2 * P2
    
    # Basis 2: Gaussian terms
    # Centers at -0.66, -0.33, 0, 0.33, 0.66
    # Width sigma = 0.5
    # G_c(t) = exp(-((t - c) / sigma)^2)
    sigma = 0.5
    c1 = -0.66
    c2 = -0.33
    c3 = 0.0
    c4 = 0.33
    c5 = 0.66
    
    G1 = np.exp(-((t - c1) / sigma) ** 2)
    G2 = np.exp(-((t - c2) / sigma) ** 2)
    G3 = np.exp(-((t - c3) / sigma) ** 2)
    G4 = np.exp(-((t - c4) / sigma) ** 2)
    G5 = np.exp(-((t - c5) / sigma) ** 2)
    
    # Combine with parameters
    # params[0..4] for Polynomials
    # params[5..9] for Gaussians
    y = (params[0] * P0 + 
         params[1] * P1 + 
         params[2] * P2 + 
         params[3] * P3 + 
         params[4] * P4 +
         params[5] * G1 + 
         params[6] * G2 + 
         params[7] * G3 + 
         params[8] * G4 + 
         params[9] * G5)
         
    return y
