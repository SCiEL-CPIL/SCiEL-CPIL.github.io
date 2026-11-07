"""Symbolic regression: nonlinear oscillator (2) - RBF network approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network for nonlinear oscillator dynamics.
    
    Uses 10 fitted constants as weights for 10 RBF basis functions
    centered at distinct points in the input space.
    
    Inputs:
        t: time, range [30, 50]
        x: position, range [-0.2056, 0.1749]
        v: velocity, range [-0.3391, 0.3624]
    """
    # Define RBF centers - spread across the input domain
    # t range: [30, 50], x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    
    # 10 centers chosen to cover the input space well
    t_centers = np.array([32.0, 36.0, 40.0, 44.0, 48.0, 33.0, 38.0, 42.0, 46.0, 35.0])
    x_centers = np.array([-0.15, 0.0, 0.15, -0.10, 0.10, -0.05, 0.05, -0.18, 0.18, 0.0])
    v_centers = np.array([-0.25, 0.0, 0.25, -0.15, 0.15, -0.08, 0.08, -0.30, 0.30, 0.0])
    
    # Widths scaled to domain size - smaller width = more local
    # t span ~20, x span ~0.38, v span ~0.70
    t_width = 4.0
    x_width = 0.12
    v_width = 0.20
    
    # Compute squared distances for each RBF
    dt = t - t_centers[None, :] if t.ndim == 1 else t - t_centers
    dx = x - x_centers[None, :] if x.ndim == 1 else x - x_centers
    dv = v - v_centers[None, :] if v.ndim == 1 else v - v_centers
    
    # Normalize distances
    dt_norm = dt / t_width
    dx_norm = dx / x_width
    dv_norm = dv / v_width
    
    # Compute RBF values: exp(-0.5 * (dt^2 + dx^2 + dv^2))
    # Each RBF i uses its own center
    # rbf[i] = exp(-0.5 * (dt[:,i]^2 + dx[:,i]^2 + dv[:,i]^2))
    
    rbf = np.exp(-0.5 * (dt_norm**2 + dx_norm**2 + dv_norm**2))
    
    # Weighted sum of RBFs
    # params has 10 values, one weight per RBF
    result = np.sum(rbf * params[None, :], axis=1)
    
    return result
# EVOLVE-BLOCK-END
