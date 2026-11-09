import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Radial Basis Functions (RBFs).
    
    The model is: y = sum_{i=1}^{10} params[i] * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    This structure provides:
    1. High local flexibility to capture non-polynomial features.
    2. Linearity in parameters, ensuring BFGS converges to the global optimum for this basis.
    3. Well-conditioned basis if centers are distinct and bandwidth is appropriate.
    """
    # Input domain is [0.1, 6]. We place 10 centers evenly across this range.
    # Adding a small margin to avoid boundary effects if the function has edges.
    x_min, x_max = 0.1, 6.0
    n_centers = 10
    
    # Centers evenly spaced
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Bandwidth (sigma). 
    # A smaller sigma makes the basis more local (sharper peaks).
    # A larger sigma makes it smoother.
    # With 10 centers over a range of ~5.9, spacing is ~0.65.
    # sigma = 0.5 gives good overlap without being too smooth.
    sigma = 0.5
    
    # Pre-compute the squared distances (x - c_i)^2
    # x shape: (N,), centers shape: (10,)
    # We want (N, 10) matrix of squared distances
    x_col = x[:, np.newaxis]  # (N, 1)
    c_row = centers[np.newaxis, :]  # (1, 10)
    
    dist_sq = (x_col - c_row) ** 2  # (N, 10)
    
    # Compute RBF values: exp(-dist_sq / (2 * sigma^2))
    # 2 * sigma^2 = 2 * 0.25 = 0.5
    rbf_values = np.exp(-dist_sq / 0.5)  # (N, 10)
    
    # Linear combination: rbf_values (N, 10) @ params (10,) -> (N,)
    return rbf_values @ params
