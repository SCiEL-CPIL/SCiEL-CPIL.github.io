import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a learned Gaussian Radial Basis Function (RBF) network.
    
    Strategy:
    - Instead of a fixed grid of basis functions (like the linear splines in the parent), 
      we allow the optimizer to learn the positions (centers) and scales (widths) of the basis functions.
    - We use 3 Gaussian kernels.
    - Parameters (10 total):
      - p[0:3]: Centers of the 3 Gaussians (c1, c2, c3)
      - p[3:6]: Widths of the 3 Gaussians (w1, w2, w3). We use softplus to ensure positive widths.
      - p[6:9]: Weights of the 3 Gaussians (a1, a2, a3)
      - p[9]: Global bias term (b)
    
    The Gaussian kernel is exp(-((x - c)^2) / (2 * w^2)).
    Using softplus for widths ensures w > 0, which keeps the function smooth and differentiable 
    with respect to the raw parameters, aiding BFGS.
    
    This approach is materially different from the fixed linear spline because:
    1. It captures smooth, non-linear features (Gaussian bumps) rather than just piecewise linear ones.
    2. The basis functions are adaptive; the optimizer can move the centers to where the function 
       has significant curvature or amplitude, rather than being stuck on a fixed grid.
    3. It is well-suited for functions that are sums of localized smooth features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    # Centers
    c1 = params[0]
    c2 = params[1]
    c3 = params[2]
    
    # Raw widths, transform to positive widths using softplus: log(1 + exp(x))
    # To avoid overflow in exp, use a stable softplus implementation
    raw_w1 = params[3]
    raw_w2 = params[4]
    raw_w3 = params[5]
    
    # Stable softplus: x + log1p(exp(-|x|))
    def softplus(z):
        return np.where(z > 20, z, np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0))
        
    w1 = softplus(raw_w1)
    w2 = softplus(raw_w2)
    w3 = softplus(raw_w3)
    
    # Weights
    a1 = params[6]
    a2 = params[7]
    a3 = params[8]
    
    # Bias
    b = params[9]
    
    # Compute Gaussians
    # Avoid division by zero if width is extremely small (though softplus > 0)
    # Minimum width guard for numerical stability
    eps = 1e-6
    w1 = np.maximum(w1, eps)
    w2 = np.maximum(w2, eps)
    w3 = np.maximum(w3, eps)
    
    # Calculate exponent terms
    # Use vectorized operations
    term1 = a1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    term2 = a2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    term3 = a3 * np.exp(-((x - c3) ** 2) / (2.0 * w3 ** 2))
    
    return term1 + term2 + term3 + b
