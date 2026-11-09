import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed basis: Chebyshev polynomials + trigonometric functions.
    Maps x from [-2, 2] to t in [-1, 1].
    Basis: [T0, T1, T2, T3, T4, cos(pi*t), sin(pi*t), cos(2*pi*t), sin(2*pi*t), T5]
    """
    t = x * 0.5  # Map to [-1, 1]
    
    # Chebyshev polynomials T_0 to T_5
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Trigonometric terms
    c1 = np.cos(np.pi * t)
    s1 = np.sin(np.pi * t)
    c2 = np.cos(2.0 * np.pi * t)
    s2 = np.sin(2.0 * np.pi * t)
    
    # Combine all 10 basis functions with params
    return (params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3 +
            params[4] * T4 + params[5] * c1 + params[6] * s1 + params[7] * c2 +
            params[8] * s2 + params[9] * T5)
