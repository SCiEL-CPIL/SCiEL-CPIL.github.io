import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t = x / 3.0
    
    # Compute Legendre Polynomials P_0 to P_6
    P0 = np.ones_like(t)
    P1 = t
    
    P = [P0, P1]
    for n in range(1, 6):
        # Calculate P_{n+1}
        P_next = ((2*n + 1) * t * P[n] - n * P[n-1]) / (n + 1)
        P.append(P_next)
        
    # P[0] to P[6] are the first 7 basis functions
    # Add smooth non-polynomial basis functions
    # sin(x), cos(x), tanh(x)
    
    b7 = np.sin(x)
    b8 = np.cos(x)
    b9 = np.tanh(x)
    
    return (params[0] * P[0] + params[1] * P[1] + params[2] * P[2] +
            params[3] * P[3] + params[4] * P[4] + params[5] * P[5] +
            params[6] * P[6] + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
