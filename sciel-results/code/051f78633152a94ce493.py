import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [0.1, 6] to [-1, 1]
    # Center = 3.05, Half-Range = 2.95
    u = (x - 3.05) / 2.95
    
    # Hybrid Basis:
    # 1. Polynomial terms (low order)
    # 2. Trigonometric terms
    # 3. Exponential terms
    
    # Ensure output is float array
    y = np.zeros_like(x, dtype=float)
    
    # Poly
    y += params[0] * 1.0
    y += params[1] * u
    y += params[2] * u**2
    y += params[3] * u**3
    
    # Trig
    y += params[4] * np.sin(np.pi * u)
    y += params[5] * np.cos(np.pi * u)
    y += params[6] * np.sin(2 * np.pi * u)
    y += params[7] * np.cos(2 * np.pi * u)
    
    # Exp
    y += params[8] * np.exp(u)
    y += params[9] * np.exp(-u)
    
    return y
