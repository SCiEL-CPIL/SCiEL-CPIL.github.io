import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width
    # Spacing is 4/9 ~ 0.44. 
    # A narrower width (e.g., 0.3) reduces collinearity between basis functions
    # and allows for better approximation of local features, while still
    # maintaining sufficient overlap for smoothness.
    sigma = 0.3
    
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
