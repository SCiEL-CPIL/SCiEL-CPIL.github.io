"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 145.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    
    Strategy: Localized Radial Basis Function (RBF) network with Gaussian kernels.
    
    This approach uses 9 fixed centers arranged in a 3x3 grid over the domain [-1, 1]^2.
    Each center has a Gaussian kernel with a fixed bandwidth (sigma=0.5).
    The parameters correspond to the weights of these 9 kernels plus a global bias.
    
    The prediction is a linear combination of the Gaussian kernels evaluated at (x, y).
    This basis is smooth and local, avoiding the severe ill-conditioning associated with
    global high-order polynomial bases (like Chebyshev or monomials) while being more
    flexible than simple piecewise-linear approximations. It allows BFGS to converge
    robustly to a local minimum that can capture complex, non-smooth, or high-frequency
    features better than low-order global polynomials.
    
    Parameters:
    p[0..8]: Weights for the 9 Gaussian kernels centered at:
             (-1, -1), (0, -1), (1, -1)
             (-1,  0), (0,  0), (1,  0)
             (-1,  1), (0,  1), (1,  1)
    p[9]:    Global constant bias.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure inputs are 1D for safety
    x = x.ravel()
    y = y.ravel()
    
    N = len(x)
    
    # Define fixed centers for the 3x3 grid
    centers_x = np.array([-1.0, 0.0, 1.0])
    centers_y = np.array([-1.0, 0.0, 1.0])
    
    # Bandwidth for the Gaussian kernels
    sigma = 0.5
    sigma_sq = 2.0 * sigma * sigma  # 2 * sigma^2 for the exponent
    
    # Compute the RBF features
    # We need to compute exp(-|| (x, y) - c_i ||^2 / (2 * sigma^2)) for each center i
    # To avoid creating a huge (N, 9) array explicitly in a slow loop, we vectorize.
    
    # Create meshgrid of centers
    cx, cy = np.meshgrid(centers_x, centers_y, indexing='ij')
    cx = cx.ravel()  # shape (9,)
    cy = cy.ravel()  # shape (9,)
    
    # Compute squared distances from each point (x, y) to each center (cx, cy)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, None] - cx[None, :]  # shape (N, 9)
    dy = y[:, None] - cy[None, :]  # shape (N, 9)
    dist_sq = dx**2 + dy**2       # shape (N, 9)
    
    # Compute Gaussian kernels
    # kernel[i, j] = exp(-dist_sq[i, j] / (2 * sigma^2))
    kernels = np.exp(-dist_sq / sigma_sq)
    
    # Prediction: weighted sum of kernels + bias
    # kernels shape (N, 9), params[:9] shape (9,)
    pred = kernels @ params[:9] + params[9]
    
    return pred

# EVOLVE-BLOCK-END
