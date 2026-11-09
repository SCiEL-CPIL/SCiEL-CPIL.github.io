import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Sum of Gaussian Radial Basis Functions.
    
    The model is: y_hat = sum_{i=1}^{9} w_i * exp(-0.5 * inv_var * ||(x, y) - c_i||^2)
    
    params:
    - params[0:9]: Weights for the 9 RBF centers.
    - params[9]: Inverse variance (1/sigma^2) of the Gaussian kernels.
    
    The centers are fixed on a 3x3 grid within [-1, 1] to provide a stable basis.
    This allows the model to capture non-polynomial, smooth variations that a cubic 
    polynomial might miss, while BFGS optimizes the weights and smoothness.
    """
    # Fixed centers on a 3x3 grid
    # x_coords, y_coords in {-0.8, 0.0, 0.8}
    cx = np.array([-0.8, 0.0, 0.8])
    cy = np.array([-0.8, 0.0, 0.8])
    
    # Create meshgrid of centers
    Cx, Cy = np.meshgrid(cx, cy, indexing='ij')
    Cx = Cx.flatten()
    Cy = Cy.flatten()
    
    # Extract parameters
    weights = params[0:9]
    inv_var = params[9]
    
    # Clamp inv_var to prevent numerical issues (sigma too small or large)
    # If inv_var is very small, kernels are very wide (polynomial-like).
    # If inv_var is very large, kernels are very narrow (spiky).
    # We allow a range of smoothness.
    inv_var = np.clip(inv_var, 0.01, 100.0)
    
    # Compute distances from each input point (x, y) to each center (Cx, Cy)
    # x, y shape: (N,)
    # Cx, Cy shape: (9,)
    # We want dist_sq shape: (N, 9)
    
    dx = x[:, np.newaxis] - Cx[np.newaxis, :]
    dy = y[:, np.newaxis] - Cy[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Compute RBF values: exp(-0.5 * inv_var * dist_sq)
    rbf_values = np.exp(-0.5 * inv_var * dist_sq)
    
    # Linear combination of RBFs
    prediction = rbf_values @ weights
    
    return prediction
