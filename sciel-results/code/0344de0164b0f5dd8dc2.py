# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_T0 = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs using a Localized Radial Basis Function (RBF) network.
    
    Structure:
    - 9 Gaussian RBFs centered on a 3x3 grid over [-1, 1]^2.
    - 1 Global Constant term.
    - Total 10 parameters.
    
    Why this structure:
    - Global polynomials (current program) suffer from ill-conditioning and struggle with local features.
    - RBFs provide local approximation capabilities.
    - The constant term ensures the model can easily capture the mean of the function,
      allowing BFGS to converge to the correct offset quickly, while the RBFs 
      adjust to the local variations.
    - Fixed centers and widths ensure the basis is well-defined and stable for optimization.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
        
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback if we are about to time out (should not happen for N=200)
        return np.zeros_like(x)

    # Define 9 centers on a 3x3 grid
    centers_x = [-1.0, 0.0, 1.0]
    centers_y = [-1.0, 0.0, 1.0]
    
    # Width (sigma) for the Gaussian kernels.
    # A sigma of 0.5 allows for some overlap between adjacent centers (distance 1.0),
    # enabling smooth interpolation while maintaining locality.
    sigma = 0.5
    sigma_sq = 2.0 * sigma * sigma
    
    # Initialize prediction with the constant term (param 0)
    # p[0] is the coefficient for the constant basis function 1
    pred = np.full_like(x, p[0], dtype=float)
    
    # Add contributions from the 9 RBFs
    # params[1:10] correspond to the 9 centers
    idx = 1
    for cx in centers_x:
        for cy in centers_y:
            # Compute squared distance
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            
            # Gaussian kernel
            k = np.exp(-r2 / sigma_sq)
            
            pred += p[idx] * k
            idx += 1
            
    # Guard against non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
# EVOLVE-BLOCK-END
