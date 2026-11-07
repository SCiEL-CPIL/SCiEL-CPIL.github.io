import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    The model is a linear combination of 9 Gaussian RBFs centered on a 3x3 grid
    within the domain [-1, 1]x[-1, 1], plus a constant term.
    
    Total parameters: 10 (9 Gaussian weights + 1 constant).
    The model is linear in params, ensuring BFGS convergence is stable and fast.
    
    RBF centers:
    Grid points: -0.66, 0.0, 0.66 for both x and y.
    Sigma (width): 0.7
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
        
    # Define RBF centers on a 3x3 grid
    # Spacing is roughly 2/3 to cover [-1, 1] well
    centers_x = np.array([-0.66, 0.0, 0.66])
    centers_y = np.array([-0.66, 0.0, 0.66])
    
    # Define the width (sigma) of the Gaussians
    # A sigma of 0.7 provides good overlap for smooth functions
    sigma = 0.7
    sigma_sq_2 = 2.0 * sigma * sigma
    
    # Precompute squared distances for all 9 centers
    # x shape: (n,), centers_x shape: (3,)
    # dist_sq shape: (n, 3) for x, (n, 3) for y
    # We need (n, 9) total squared distances
    
    # Vectorized distance calculation
    # x[:, None] is (n, 1), centers_x[None, :] is (1, 3) -> diff is (n, 3)
    dx = x[:, None] - centers_x[None, :]
    dy = y[:, None] - centers_y[None, :]
    
    # Expand to 9 centers by combining x and y differences
    # We want a matrix of shape (n, 9) where each column corresponds to a center (cx, cy)
    # Center order:
    # 0: (-0.66, -0.66)
    # 1: (-0.66, 0.0)
    # 2: (-0.66, 0.66)
    # 3: (0.0, -0.66)
    # 4: (0.0, 0.0)
    # 5: (0.0, 0.66)
    # 6: (0.66, -0.66)
    # 7: (0.66, 0.0)
    # 8: (0.66, 0.66)
    
    # dx is (n, 3), dy is (n, 3)
    # We can construct the 9 basis columns by iterating or using meshgrid-like logic
    # For speed and clarity, we can do:
    basis = np.zeros((n, 9), dtype=np.float64)
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            # dist_sq = (x - cx_i)^2 + (y - cy_j)^2
            dist_sq = dx[:, i]**2 + dy[:, j]**2
            basis[:, idx] = np.exp(-dist_sq / sigma_sq_2)
            
    # Construct the full design matrix (n, 10)
    # Column 0 is the constant term (1)
    # Columns 1-9 are the RBF outputs
    design_matrix = np.column_stack([np.ones(n), basis])
    
    # Linear combination
    out = design_matrix @ p
    
    # Safety check
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
    return out
