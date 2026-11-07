"""Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a sum of Gaussian basis functions.
    
    Parameters:
        t: time (input 1)
        x: position (input 2)
        v: velocity (input 3)
        params: array of 10 parameters.
                params[0:3] = center 1 (t_c1, x_c1, v_c1)
                params[3]   = weight 1 (w1)
                params[4:7] = center 2 (t_c2, x_c2, v_c2)
                params[7]   = weight 2 (w2)
                params[8]   = shared scale (inverse bandwidth)
                params[9]   = bias (offset)
    
    Returns:
        Acceleration predictions (same shape as t)
    """
    # Extract parameters
    t_c1, x_c1, v_c1 = params[0], params[1], params[2]
    w1 = params[3]
    
    t_c2, x_c2, v_c2 = params[4], params[5], params[6]
    w2 = params[7]
    
    scale = params[8]
    bias = params[9]
    
    # Ensure scale is positive and not too small to avoid numerical issues
    # Use a lower bound to prevent division by zero or extreme flatness
    scale = np.maximum(np.abs(scale), 1e-4)
    
    # Compute Gaussian basis functions
    # g(t, x, v) = exp(-scale * ((t - t_c)^2 + (x - x_c)^2 + (v - v_c)^2))
    
    dt1 = t - t_c1
    dx1 = x - x_c1
    dv1 = v - v_c1
    r2_1 = dt1**2 + dx1**2 + dv1**2
    g1 = np.exp(-scale * r2_1)
    
    dt2 = t - t_c2
    dx2 = x - x_c2
    dv2 = v - v_c2
    r2_2 = dt2**2 + dx2**2 + dv2**2
    g2 = np.exp(-scale * r2_2)
    
    # Weighted sum plus bias
    # The bias term allows the model to fit a non-zero mean acceleration,
    # which is crucial for many oscillator dynamics and improves baseline performance.
    
    return bias + w1 * g1 + w2 * g2
# EVOLVE-BLOCK-END
