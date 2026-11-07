import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Gaussian Radial Basis Functions (RBF)
    # This uses local basis functions centered at fixed points in the domain.
    # Unlike global polynomials, RBFs can model local features and sharp changes.
    
    # Fixed centers for the 10 Gaussian basis functions
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width (standard deviation) for the Gaussians
    # A width of 0.75 covers a reasonable range, allowing overlap but maintaining locality
    sigma = 0.75
    beta = 1.0 / (2.0 * sigma * sigma)
    
    # Compute basis matrix
    # x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Basis values
    B = np.exp(-beta * diff * diff)
    
    # Linear combination
    y = B @ params
    
    return y
