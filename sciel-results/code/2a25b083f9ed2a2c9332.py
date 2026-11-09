import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a combination of:
    - Linear trend
    - Polynomial basis (up to degree 5)
    - Trigonometric terms (sin, cos at multiple frequencies)
    - Gaussian RBFs for local features
    - Tanh for saturating behavior
    
    params: array of length 10
      params[0]: linear slope
      params[1]: linear intercept
      params[2]: quadratic coefficient
      params[3]: cubic coefficient
      params[4]: sin frequency amplitude
      params[5]: cos frequency amplitude
      params[6]: tanh amplitude
      params[7]: RBF amplitude 1
      params[8]: RBF amplitude 2
      params[9]: RBF amplitude 3
    """
    # Scale x to [-1, 1] for numerical stability
    t = x / 3.0
    
    # Linear trend
    y_linear = params[0] * x + params[1]
    
    # Polynomial terms (scaled)
    y_poly = params[2] * t**2 + params[3] * t**3
    
    # Trigonometric terms
    y_trig = params[4] * np.sin(2.0 * np.pi * t) + params[5] * np.cos(4.0 * np.pi * t)
    
    # Tanh for saturating behavior
    y_tanh = params[6] * np.tanh(2.0 * t)
    
    # Gaussian RBFs at fixed centers
    centers = np.array([-1.5, 0.0, 1.5])
    sigma = 0.8
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma)**2)
    y_rbf = params[7] * rbf_values[:, 0] + params[8] * rbf_values[:, 1] + params[9] * rbf_values[:, 2]
    
    # Combine all components
    y = y_linear + y_poly + y_trig + y_tanh + y_rbf
    
    return y
