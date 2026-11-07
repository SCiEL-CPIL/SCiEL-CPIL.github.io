import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Extract parameters
    # Params 0-8: Weights for 9 Gaussian RBFs
    # Param 9: Bias term
    weights = params[:9]
    bias = params[9]
    
    # Fixed centers covering the domain [-2, 2]
    # 9 centers evenly spaced
    centers = np.linspace(-2.0, 2.0, 9)
    
    # Fixed width for RBFs
    # Chosen to provide sufficient overlap for smooth approximation
    sigma = 0.4
    
    # Compute RBF terms
    # (x[:, None] - centers[None, :]) creates a matrix of differences
    # This is efficient for vectorized operations
    diff = x[:, None] - centers[None, :]
    
    # Calculate Gaussian values: exp(-0.5 * (diff/sigma)^2)
    # Using 0.5 * (diff/sigma)^2 is equivalent to (diff^2) / (2*sigma^2)
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of RBFs plus bias
    # rbf_values @ weights performs the dot product
    return rbf_values @ weights + bias
