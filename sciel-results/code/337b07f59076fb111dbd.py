import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] to improve conditioning for polynomial terms
    u = x / 3.0
    
    # Basis functions
    b0 = np.ones_like(u)
    b1 = u
    b2 = u ** 2
    b3 = u ** 3
    b4 = u ** 4
    b5 = np.sin(np.pi * u)
    b6 = np.cos(np.pi * u)
    b7 = np.sin(2 * np.pi * u)
    b8 = np.exp(-u ** 2)
    b9 = np.tanh(u)
    
    # Combine linearly
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
