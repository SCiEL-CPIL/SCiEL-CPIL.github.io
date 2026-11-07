import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Gaussian RBFs.
    
    params: array of length 10.
    params[0:9] are weights for 9 RBFs.
    params[9] is the bias term.
    """
    # Extract parameters
    weights = params[:9]
    bias = params[9]
    
    # Fixed centers covering the domain [-2, 2]
    centers = np.linspace(-2.0, 2.0, 9)
    
    # Fixed width for the Gaussian kernels
    # sigma=0.5 provides good local resolution while maintaining smoothness
    sigma = 0.5
    
    # Compute the RBF basis matrix
    # x has shape (N,), centers has shape (9,)
    # diff has shape (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF values: exp(-0.5 * (diff/sigma)^2)
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    prediction = rbf_values @ weights + bias
    
    return prediction
