import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis of Fourier series and polynomials.
    
    The basis is designed to be well-conditioned for BFGS optimization starting from ones.
    Input x is in [-3, 3]. We scale to t in [-1, 1].
    
    Parameters:
    p[0]: DC offset
    p[1]: Amp sin(pi t)
    p[2]: Amp cos(pi t)
    p[3]: Amp sin(2 pi t)
    p[4]: Amp cos(2 pi t)
    p[5]: Amp sin(3 pi t)
    p[6]: Amp cos(3 pi t)
    p[7]: Linear coeff (t)
    p[8]: Quadratic coeff (t^2)
    p[9]: Cubic coeff (t^3)
    """
    # Scale input to [-1, 1]
    t = x / 3.0
    
    # Pre-compute basis functions
    # Trig terms
    # Using np.pi for accurate frequencies
    sin_1 = np.sin(np.pi * t)
    cos_1 = np.cos(np.pi * t)
    sin_2 = np.sin(2.0 * np.pi * t)
    cos_2 = np.cos(2.0 * np.pi * t)
    sin_3 = np.sin(3.0 * np.pi * t)
    cos_3 = np.cos(3.0 * np.pi * t)
    
    # Polynomial terms
    # t^0 is 1, but we have a separate DC param p[0] which acts as intercept.
    # To avoid collinearity between p[0] and p[8] (t^2), we keep them distinct.
    # t^1, t^2, t^3
    t_1 = t
    t_2 = t * t
    t_3 = t_2 * t
    
    # Combine linearly
    # y = p[0] + p[1]*sin_1 + p[2]*cos_1 + ...
    y = np.full_like(t, params[0], dtype=float)
    y += params[1] * sin_1
    y += params[2] * cos_1
    y += params[3] * sin_2
    y += params[4] * cos_2
    y += params[5] * sin_3
    y += params[6] * cos_3
    y += params[7] * t_1
    y += params[8] * t_2
    y += params[9] * t_3
    
    return y
