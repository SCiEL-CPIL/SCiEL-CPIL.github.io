"""Symbolic regression for a nonlinear oscillator using RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135  # 15s buffer before 150s kill

def _time_left() -> float:
    return _DEADLINE - time.time()

# Fixed Radial Basis Function parameters
# Input ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
# We define 10 centers in a grid pattern to cover the domain.
# The RBF basis functions are: exp(-||[x, v] - c_i||^2 / (2 * sigma^2))
# Since we have 10 parameters, we use 10 centers.

# Grid points for centers
# x centers: 5 points
x_centers = np.linspace(-0.55, 0.5, 5)
# v centers: 2 points (since we only have 10 params total, 5x2 grid)
v_centers = np.linspace(-0.15, 0.25, 2)

# Create meshgrid of centers
CX, CV = np.meshgrid(x_centers, v_centers)
CX = CX.flatten()
CV = CV.flatten()

# Ensure we have exactly 10 centers
if len(CX) > 10:
    CX = CX[:10]
    CV = CV[:10]
elif len(CX) < 10:
    # If fewer, pad with zeros (should not happen with 5x2)
    pass

# Width of the RBFs. 
# If sigma is too small, the basis is very local and hard to fit with only 10 weights if they don't align perfectly.
# If sigma is too large, it behaves like a global linear model.
# Input range width is approx 1.1 for x and 0.45 for v.
# A sigma around 0.3-0.5 seems reasonable to allow some overlap but maintain locality.
SIGMA = 0.4
SIGMA2 = 2 * SIGMA**2

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration prediction using a sum of Radial Basis Functions.
    
    Args:
        x: position array
        v: velocity array  
        params: 10-element array of fitted constants (weights for each RBF)
    
    Returns:
        Array of acceleration predictions (one per row)
    """
    # x and v are arrays of shape (N,)
    # We need to compute distance from each point (x[i], v[i]) to each center (CX[j], CV[j])
    
    # Vectorized computation:
    # dx shape: (N, 1) - (1, 10) -> (N, 10)
    dx = x[:, np.newaxis] - CX[np.newaxis, :]
    dv = v[:, np.newaxis] - CV[np.newaxis, :]
    
    # Squared distance
    dist_sq = dx**2 + dv**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 10)
    rbf_values = np.exp(-dist_sq / SIGMA2)
    
    # Weighted sum
    # rbf_values (N, 10) dot params (10,) -> (N,)
    result = rbf_values @ params
    
    return result

# The evaluator will call equation(x, v, params) after fitting params.
# The fitting is done by the evaluator using scipy.optimize.minimize.
# We just need to provide the equation function.
# No internal fitting is required by us, as the evaluator handles it.
# However, we must ensure the function is fast and stable.

# EVOLVE-BLOCK-END
