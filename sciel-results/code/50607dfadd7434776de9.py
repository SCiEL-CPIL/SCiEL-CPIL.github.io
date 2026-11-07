import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Polynomial terms
    const = p[0]
    lin = p[1] * x
    quad = p[2] * x**2
    cubic = p[3] * x**3
    
    # Fourier terms
    sin1 = p[4] * np.sin(x)
    cos1 = p[5] * np.cos(x)
    sin2 = p[6] * np.sin(2.0 * x)
    cos2 = p[7] * np.cos(2.0 * x)
    sin3 = p[8] * np.sin(3.0 * x)
    cos3 = p[9] * np.cos(3.0 * x)
    
    return const + lin + quad + cubic + sin1 + cos1 + sin2 + cos2 + sin3 + cos3
