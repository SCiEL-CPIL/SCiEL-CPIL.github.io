import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    We use 10 fixed centers evenly spaced in [-2, 2].
    The 10 parameters are the amplitudes of the RBFs.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Define 10 fixed centers in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Define a fixed width (sigma)
    # A smaller sigma makes the RBFs more localized.
    # With 10 points in [-2, 2], spacing is ~0.44.
    # sigma = 0.5 seems reasonable.
    sigma = 0.5
    
    # Compute RBFs
    # x shape: (N, 1) or (N,)
    # centers shape: (10,)
    # We want to compute exp(-((x - c)^2) / (2 * sigma^2)) for each x and c
    
    # Reshape x to (N, 1) and centers to (1, 10) for broadcasting
    x_col = x.reshape(-1, 1)
    centers_row = centers.reshape(1, -1)
    
    # Squared distances
    sq_dist = (x_col - centers_row) ** 2
    
    # RBF values
    rbf_values = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Linear combination of RBFs with parameters
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    result = rbf_values @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
