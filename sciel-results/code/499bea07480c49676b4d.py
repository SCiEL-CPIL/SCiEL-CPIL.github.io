import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev polynomials
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_6
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    # T_5(t) = 16t^5 - 20t^3 + 5t
    # T_6(t) = 32t^6 - 48t^4 + 18t^2 - 1
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    t6 = t4 * t2
    
    b0 = np.ones_like(t)
    b1 = t
    b2 = 2.0 * t2 - 1.0
    b3 = 4.0 * t3 - 3.0 * t
    b4 = 8.0 * t4 - 8.0 * t2 + 1.0
    b5 = 16.0 * t5 - 20.0 * t3 + 5.0 * t
    b6 = 32.0 * t6 - 48.0 * t4 + 18.0 * t2 - 1.0
    
    # Additional basis functions for non-polynomial shapes
    # Using period 2*pi scaled to fit [-2,2] nicely? 
    # sin(pi * t) has period 2 in t, so period 4 in x.
    # cos(pi * t)
    # tanh(x) for saturation
    # exp(-x^2) for gaussian bump
    
    b7 = np.sin(np.pi * t)
    b8 = np.cos(np.pi * t)
    b9 = np.tanh(x)
    
    # Linear combination
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + 
         p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
         
    return y
