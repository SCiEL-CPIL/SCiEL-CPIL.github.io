import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Chebyshev polynomial (degree 3) + 3 Gaussian RBFs.
    Parameters:
    p[0:4] : Chebyshev coefficients c0, c1, c2, c3
    p[4:10]: RBF parameters (amp1, width_log1, amp2, width_log2, amp3, width_log3)
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    c0, c1, c2, c3 = params[0:4]
    a1, lw1, a2, lw2, a3, lw3 = params[4:10]
    
    # Scale x to [0, 1] for Chebyshev polynomials
    # Domain is [0.1, 6.0]
    x_min, x_max = 0.1, 6.0
    u = (x - x_min) / (x_max - x_min)
    
    # Chebyshev polynomials T_k(u)
    # T_0(u) = 1
    # T_1(u) = u
    # T_2(u) = 2u^2 - 1
    # T_3(u) = 4u^3 - 3u
    
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2 * u**2 - 1
    T3 = 4 * u**3 - 3 * u
    
    poly_part = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    
    # RBFs
    # Widths are exp(lw) to ensure positivity
    # Clip lw to avoid overflow in exp
    lw1 = np.clip(lw1, -10, 10)
    lw2 = np.clip(lw2, -10, 10)
    lw3 = np.clip(lw3, -10, 10)
    
    w1 = np.exp(lw1)
    w2 = np.exp(lw2)
    w3 = np.exp(lw3)
    
    # Fixed centers
    c1_x = 1.0
    c2_x = 3.0
    c3_x = 5.0
    
    rbf1 = a1 * np.exp(-w1 * (x - c1_x)**2)
    rbf2 = a2 * np.exp(-w2 * (x - c2_x)**2)
    rbf3 = a3 * np.exp(-w3 * (x - c3_x)**2)
    
    return poly_part + rbf1 + rbf2 + rbf3
