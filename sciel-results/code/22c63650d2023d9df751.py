"""Symbolic regression: nonlinear oscillator acceleration with RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# Precompute RBF centers and widths based on input ranges
# x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
# We'll use a grid of centers in the (x, v) space

# Define centers on a 3x3 grid in the input domain
_x_min, _x_max = -0.2056, 0.1749
_v_min, _v_max = -0.3391, 0.3624

# Create 9 centers on a 3x3 grid
_x_centers = np.linspace(_x_min, _x_max, 3)
_v_centers = np.linspace(_v_min, _v_max, 3)
_xc, _vc = np.meshgrid(_x_centers, _v_centers, indexing='ij')
_centers = np.column_stack([_xc.ravel(), _vc.ravel()])  # shape (9, 2)

# Compute bandwidth: distance between adjacent grid points
_dx = (_x_max - _x_min) / 2
_dv = (_v_max - _v_min) / 2
_bandwidth = np.sqrt(_dx**2 + _dv**2) * 0.7  # slightly smaller for more local behavior

# Precompute nothing else; centers and bandwidth are constants


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF expansion model for nonlinear oscillator acceleration.
    Uses 9 Gaussian RBFs centered on a 3x3 grid in (x, v) space.
    9 RBF weights + 1 global linear term in x = 10 parameters total.
    
    params[0:9] = RBF weights
    params[9]   = coefficient for a linear x term (to capture restoring force)
    """
    p = params  # shape (10,)
    
    # Compute RBF terms: exp(-||[x, v] - center_i||^2 / (2 * bandwidth^2))
    # x shape (N,), v shape (N,)
    # centers shape (9, 2)
    
    # Compute squared distances from each (x_i, v_i) to each center
    # dx shape (N, 9), dv shape (N, 9)
    dx = x[:, np.newaxis] - _centers[:, 0][np.newaxis, :]  # (N, 9)
    dv = v[:, np.newaxis] - _centers[:, 1][np.newaxis, :]  # (N, 9)
    dist2 = dx * dx + dv * dv  # (N, 9)
    
    # Gaussian RBF values
    rbf_vals = np.exp(-dist2 / (2.0 * _bandwidth * _bandwidth))  # (N, 9)
    
    # Weighted sum of RBFs
    # rbf_vals shape (N, 9), p[:9] shape (9,)
    rbf_sum = rbf_vals @ p[:9]  # (N,)
    
    # Add linear x term
    linear_term = p[9] * x  # (N,)
    
    return rbf_sum + linear_term
# EVOLVE-BLOCK-END
