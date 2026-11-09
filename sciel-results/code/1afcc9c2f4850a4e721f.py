"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local linearization model using Gaussian RBFs on (x, v) phase plane.
    
    Strategy:
    Each of 10 Gaussian RBF centers on the (x, v) phase plane is multiplied
    by a local linear term (1 + a*x + b*v) to capture nonlinear coupling
    locally. The 10 parameters are the weights of these basis functions.
    
    This structure approximates the acceleration field as a piecewise-linear
    function in the phase plane, with smooth transitions between regions.
    It captures non-additive x-v coupling more effectively than global
    polynomials while keeping only 10 free parameters.
    
    Input Ranges:
    x: [-0.2056, 0.1749]
    v: [-0.3391, 0.3624]
    
    Centers: 5x2 grid on (x, v) in normalized [0,1] space.
    """
    
    # Normalization constants
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    x_norm = (x - x_min) / (x_max - x_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # Define 10 centers in normalized (x, v) space: 5 x 2 grid
    x_centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_centers = np.array([0.25, 0.75])
    
    # Create all 10 center combinations
    cx, cv = np.meshgrid(x_centers, v_centers, indexing='ij')
    centers_x = cx.flatten()  # (10,)
    centers_v = cv.flatten()  # (10,)
    
    # Compute Gaussian kernel values: (N, 10)
    dx = x_norm[:, np.newaxis] - centers_x[np.newaxis, :]
    dv = v_norm[:, np.newaxis] - centers_v[np.newaxis, :]
    sq_dist = dx**2 + dv**2
    
    # Gaussian kernel with sigma tuned for local resolution
    sigma = 0.25
    kernel_vals = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Local linear term: (1 + x_norm + v_norm) per basis function
    # This gives each RBF a local linear slope to capture phase-plane dynamics
    linear_term = 1.0 + x_norm[:, np.newaxis] + v_norm[:, np.newaxis]
    
    # Basis matrix: kernel * linear_term, shape (N, 10)
    basis = kernel_vals * linear_term
    
    # Linear combination with 10 parameters
    return basis @ params

# EVOLVE-BLOCK-END
