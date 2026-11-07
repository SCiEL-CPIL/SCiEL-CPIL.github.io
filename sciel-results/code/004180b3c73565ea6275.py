import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a linear combination of Gaussian RBFs in x.
    
    x: input array (values in [0.1, 6])
    params: array of length 10, weights for 10 Gaussian RBFs
    
    Returns: array of predictions, one per point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    if n == 0:
        return np.empty(0, dtype=np.float64)
    
    # Define 10 evenly spaced centers across [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed widths: use a width that covers roughly 1/9 of the domain
    # This gives good local coverage with some overlap
    width = (x_max - x_min) / 9.0 * 0.7
    
    # Compute RBF values: sum_i params[i] * exp(-((x - centers[i])^2) / (2*width^2))
    # Vectorized computation
    # x shape: (n,), centers shape: (10,)
    # diff shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian RBF
    rbf_values = np.exp(-diff ** 2 / (2.0 * width ** 2))
    
    # Linear combination
    result = rbf_values @ params
    
    return result
