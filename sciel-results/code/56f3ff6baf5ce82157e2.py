import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a numpy array of floats
    x = np.asarray(x, dtype=float)
    
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    u = (x - 3.05) / 2.95
    
    # Define a hybrid basis combining:
    # 1. Low-order Chebyshev polynomials (well-conditioned for smooth parts)
    # 2. Exponential terms (for growth/decay behavior)
    # 3. Trigonometric terms (for periodic behavior)
    
    # Chebyshev polynomials T0 to T3
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2 * u * T1 - T0  # 2u^2 - 1
    T3 = 2 * u * T2 - T1  # 4u^3 - 3u
    
    # Exponential terms
    E1 = np.exp(u)
    E2 = np.exp(-u)
    
    # Trigonometric terms
    S1 = np.sin(u)
    C1 = np.cos(u)
    
    # Higher frequency trigonometric
    S2 = np.sin(2 * u)
    C2 = np.cos(2 * u)
    
    # Combine basis functions with parameters
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * E1 + 
         params[5] * E2 + 
         params[6] * S1 + 
         params[7] * C1 + 
         params[8] * S2 + 
         params[9] * C2)
         
    return y
