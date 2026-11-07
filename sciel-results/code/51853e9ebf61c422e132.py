import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers for Gaussian RBFs
    # 10 centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for Gaussian kernels
    # sigma = 0.8 provides good coverage and smoothness
    sigma = 0.8
    
    # Compute Gaussian basis functions
    # x shape: (N,)
    # centers shape: (10,)
    # We need to compute exp(-((x - c_i)^2) / (2 * sigma^2)) for each i
    
    # Efficient vectorized computation
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian values
    # phi shape: (N, 10)
    phi = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: sum(params[i] * phi[i])
    # params shape: (10,)
    # result shape: (N,)
    return phi @ params
