"""
Symbolic regression for a nonlinear oscillator using a Gaussian Radial Basis Function (RBF) basis.
Strategy: Use 10 fixed Gaussian centers distributed across the input domain (x, v).
The equation is a linear combination of these RBFs. This transforms the problem into
a linear regression problem with respect to the parameters, which BFGS solves very efficiently
and robustly, avoiding the local minima traps of non-linear parameterization.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 130

def _time_left() -> float:
    return _DEADLINE - time.time()

# Pre-compute RBF centers and widths
# Input ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
# We create a 3x3 grid of centers (9 centers) + 1 global constant term = 10 parameters
# This ensures coverage of the entire input domain.

# Center grid
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

# Create 3x3 grid
x_centers = np.linspace(x_min, x_max, 3)
v_centers = np.linspace(v_min, v_max, 3)

# Flatten grid to 9 points
centers = np.array([(x, v) for x in x_centers for v in v_centers])

# Width (sigma) of the Gaussians. 
# The domain size in x is ~1.1, in v is ~0.45.
# A sigma of ~0.3-0.5 should provide good overlap and smoothness.
# Let's use sigma_x = 0.4 and sigma_v = 0.2 to match aspect ratio roughly.
# Actually, let's use a single isotropic sigma for simplicity, scaled to the domain.
# Domain diagonal is approx sqrt(1.1^2 + 0.45^2) ~ 1.19.
# Sigma = 0.3 seems reasonable.
SIGMA_X = 0.4
SIGMA_V = 0.2

# Pre-compute center coordinates for vectorized operations
C_X = centers[:, 0]  # shape (9,)
C_V = centers[:, 1]  # shape (9,)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target using a linear combination of Gaussian RBFs.
    
    Args:
        x: Position array (N,)
        v: Velocity array (N,)
        params: Array of 10 parameters. 
                params[0:9] are weights for the 9 RBFs.
                params[9] is a global bias term.
    
    Returns:
        Prediction array (N,)
    """
    try:
        # Reshape inputs for broadcasting: (N, 1)
        x_col = x.reshape(-1, 1)
        v_col = v.reshape(-1, 1)
        
        # Compute squared distances to each center
        # dx: (N, 9), dv: (N, 9)
        dx = x_col - C_X  # (N, 9)
        dv = v_col - C_V  # (N, 9)
        
        # Squared distance: (N, 9)
        dist_sq = (dx ** 2) / (SIGMA_X ** 2) + (dv ** 2) / (SIGMA_V ** 2)
        
        # Gaussian basis values: (N, 9)
        basis = np.exp(-0.5 * dist_sq)
        
        # Linear combination: (N, 9) @ (9,) -> (N,)
        # params[0:9] are the RBF weights
        pred = basis @ params[0:9]
        
        # Add global bias
        pred += params[9]
        
        # Ensure finite output
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
        
    except Exception:
        # Fallback to simple linear if something goes wrong (should not happen with RBFs)
        try:
            return params[0]*x + params[1]*v + params[2]
        except Exception:
            return np.zeros_like(x)

# EVOLVE-BLOCK-END
