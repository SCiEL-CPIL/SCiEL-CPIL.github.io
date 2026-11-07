import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev + Gaussian RBF basis.
    params[0:4] are coefficients for Chebyshev polynomials T_0 to T_3.
    params[4:10] are coefficients for 6 Gaussian RBFs at fixed centers.
    """
    if len(params) == 0:
        return np.zeros_like(x)
    
    # 1. Chebyshev Polynomials part
    # Scale x to [-1, 1] for standard Chebyshev definition
    u = x / 2.0
    
    # T_0 = 1
    t0 = np.ones_like(u)
    # T_1 = u
    t1 = u
    # T_2 = 2*u^2 - 1
    t2 = 2 * u * u - 1
    # T_3 = 4*u^3 - 3*u
    t3 = 4 * u**3 - 3 * u
    
    res = params[0] * t0
    if len(params) > 1:
        res += params[1] * t1
    if len(params) > 2:
        res += params[2] * t2
    if len(params) > 3:
        res += params[3] * t3
    
    # 2. Gaussian RBFs part
    # Fixed centers spread across the domain [-2, 2]
    centers = np.array([-1.5, -0.75, -0.25, 0.25, 0.75, 1.5])
    sigma = 0.5  # Fixed width
    
    if len(params) > 4:
        for j in range(6):
            if 4 + j >= len(params):
                break
            c = centers[j]
            # Gaussian: exp(-0.5 * ((x - c) / sigma)^2)
            g = np.exp(-0.5 * ((x - c) / sigma) ** 2)
            res += params[4 + j] * g
            
    return res
