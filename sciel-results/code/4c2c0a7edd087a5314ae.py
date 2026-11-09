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
    
    Unlike the global bilinear interpolation, this approach uses 9 Gaussian kernels
    centered at a 3x3 grid of points in the domain [-1, 1]^2. Each kernel is localized,
    allowing the model to fit complex, non-smooth, or high-frequency local structures
    without the constraints of global linearity. The last parameter acts as a global
    bias term.
    
    The 10 parameters are:
    p[0..8]: Weights for the 9 Gaussian kernels.
             Centers are at (-1, -1), (0, -1), (1, -1)
                       (-1,  0), (0,  0), (1,  0)
                       (-1,  1), (0,  1), (1,  1)
    p[9]:    Global constant offset (bias).
    
    The prediction is the weighted sum of the Gaussian kernels plus the bias.
    This is a linear function of the parameters, making it well-suited for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure inputs are 1D for safety
    x = x.ravel()
    y = y.ravel()
    N = len(x)
    
    # Define the 3x3 grid of centers
    # Centers: (-1, -1), (0, -1), (1, -1)
    #          (-1,  0), (0,  0), (1,  0)
    #          (-1,  1), (0,  1), (1,  1)
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0]
    ])
    
    # Gaussian width parameter (sigma)
    # A smaller sigma makes the kernels more localized.
    # With centers spaced by 1.0, a sigma of 0.5 provides good overlap for smooth functions
    # but allows for local variation. 0.4 is a bit tighter.
    sigma = 0.45
    
    # Compute distances from each input point to each center
    # x is (N,), centers is (9, 2)
    # dx = x[:, None] - centers[None, :, 0] -> (N, 9)
    # dy = y[:, None] - centers[None, :, 1] -> (N, 9)
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    
    # Squared distances
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernels: exp(-dist_sq / (2 * sigma^2))
    # To avoid underflow/overflow issues, we compute the exponent carefully.
    # 2 * sigma^2 = 2 * 0.45^2 = 2 * 0.2025 = 0.405
    inv_2_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    exponents = -dist_sq * inv_2_sigma_sq
    
    # Compute kernels
    kernels = np.exp(exponents)
    
    # Prediction: weighted sum of kernels + bias
    # weights @ kernels.T + bias
    # params[:9] is (9,), kernels is (N, 9)
    # pred = kernels @ params[:9] + params[9]
    
    pred = kernels @ params[:9] + params[9]
    
    return pred

# EVOLVE-BLOCK-END
