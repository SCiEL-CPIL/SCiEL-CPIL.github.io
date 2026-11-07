import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width
    # Adjusted width to allow for more detailed fitting of potential high-frequency components.
    # Smaller sigma increases the locality of each basis function.
    sigma = 0.35
    
    # Calculate Gaussian basis functions
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent term
    # exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination with params
    # basis shape: (N, 10)
    # params shape: (10,)
    predictions = basis @ params
    
    return predictions
