import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) network.
    
    Strategy:
    - Uses 9 fixed Gaussian centers arranged in a 3x3 grid over [-1, 1].
    - The 10 parameters correspond to the weights of these 9 centers plus a global bias.
    - Gaussian RBFs provide local flexibility, allowing the model to capture 
      non-polynomial features and sharp variations better than global polynomials.
    - The width (sigma) is fixed to 0.5 to balance locality and smoothness.
    - This structure is highly flexible for BFGS optimization as the loss surface 
      for linear combinations of RBFs is convex.
    """
    p = params
    
    # Define 9 fixed centers in a 3x3 grid
    # Coordinates: -1, 0, 1
    centers_x = np.array([-1.0, 0.0, 1.0])
    centers_y = np.array([-1.0, 0.0, 1.0])
    
    # Generate grid of centers
    cx, cy = np.meshgrid(centers_x, centers_y)
    cx = cx.ravel() # shape (9,)
    cy = cy.ravel() # shape (9,)
    
    # Fixed width for Gaussians
    sigma = 0.5
    
    # Precompute squared distances from each data point to each center
    # x, y are vectors of length N (number of points)
    # We want a matrix D of shape (N, 9) where D[i, j] = (x[i]-cx[j])^2 + (y[i]-cy[j])^2
    
    # x[:, None] has shape (N, 1)
    # cx[None, :] has shape (1, 9)
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    
    dist_sq = dx**2 + dy**2
    
    # Compute Gaussian basis values: exp(-dist_sq / (2 * sigma^2))
    # Factor: 1 / (2 * sigma^2) = 1 / (2 * 0.25) = 2.0
    basis = np.exp(-2.0 * dist_sq)
    
    # basis has shape (N, 9)
    # params[0:9] are weights for the 9 centers
    # params[9] is the bias
    
    # Prediction = basis @ weights + bias
    pred = basis @ p[:9] + p[9]
    
    return pred
