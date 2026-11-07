import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers evenly spaced in [-2, 2]
    # Using 10 centers to match the parameter count
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width
    # A wider sigma (1.2) ensures significant overlap between basis functions.
    # This improves the conditioning of the design matrix, helping BFGS converge
    # more reliably. It also allows the model to better capture smooth, global
    # trends (like polynomials or low-frequency sines) which are common in
    # symbolic regression benchmarks.
    sigma = 1.2
    
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
