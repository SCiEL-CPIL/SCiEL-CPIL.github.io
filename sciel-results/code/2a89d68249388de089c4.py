import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: Chebyshev polynomial (degree 3) + 2 adaptive Gaussian bumps
    # params: [c0, c1, c2, c3, mu1_raw, log_sigma1_raw, amp1, mu2_raw, log_sigma2_raw, amp2]
    
    # 1. Chebyshev Polynomial Part
    # Scale x to [-1, 1] for standard Chebyshev polynomials
    t = x / 3.0
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    
    poly = params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3
    
    # 2. Gaussian 1
    # Smoothly map raw params to constrained ranges to aid optimization
    mu1 = 3.0 * np.tanh(params[4])
    log_sigma1 = 3.0 * np.tanh(params[5]) # sigma in [exp(-3), exp(3)] ~ [0.05, 20]
    sigma1 = np.exp(log_sigma1)
    z1 = (x - mu1) / sigma1
    gauss1 = params[6] * np.exp(-0.5 * z1**2)
    
    # 3. Gaussian 2
    mu2 = 3.0 * np.tanh(params[7])
    log_sigma2 = 3.0 * np.tanh(params[8])
    sigma2 = np.exp(log_sigma2)
    z2 = (x - mu2) / sigma2
    gauss2 = params[9] * np.exp(-0.5 * z2**2)
    
    return poly + gauss1 + gauss2
