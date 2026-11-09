import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a global linear trend and
    localized Gaussian Radial Basis Functions (RBFs).
    
    This design leverages the universal approximation capability of RBFs
    to capture sharp features and non-smooth transitions that global
    polynomial or spectral expansions might miss.
    
    params: array of length 10
      params[0]: linear slope
      params[1]: linear intercept
      params[2..9]: amplitudes for 8 Gaussian RBFs
    """
    # Linear trend
    y_linear = params[0] * x + params[1]
    
    # Gaussian RBFs at evenly spaced centers in [-3, 3]
    # 8 centers distributed uniformly
    centers = np.linspace(-3.0, 3.0, 8)
    sigma = 1.0  # Width tuned for balance between local detail and smoothness
    
    # Compute RBF values: exp(-0.5 * ((x - c) / sigma)^2)
    # x shape: (N, 1), centers shape: (1, 8) -> diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma)**2)
    
    # Weighted sum of RBFs
    # params[2:] shape: (8, 1) or (8,) -> broadcast to (N, 8)
    y_rbf = np.sum(rbf_values * params[2:], axis=1)
    
    # Combine
    y = y_linear + y_rbf
    
    return y
