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
        params: array of 10 parameters defining centers, scales, and weights for 3 Gaussian bumps.
                params[0:3]  = centers for bump 1 (t_c1, x_c1, v_c1)
                params[3:6]  = centers for bump 2 (t_c2, x_c2, v_c2)
                params[6:9]  = centers for bump 3 (t_c3, x_c3, v_c3)
                params[9]    = shared scale (inverse bandwidth)
    
    Returns:
        Acceleration predictions (same shape as t)
    """
    # Extract parameters
    t_c1, x_c1, v_c1 = params[0], params[1], params[2]
    t_c2, x_c2, v_c2 = params[3], params[4], params[5]
    t_c3, x_c3, v_c3 = params[6], params[7], params[8]
    scale = params[9]
    
    # Ensure scale is positive and not too small to avoid numerical issues
    # Use softplus-like transformation to keep scale positive
    scale = np.maximum(np.abs(scale), 1e-3)
    
    # Compute Gaussian basis functions
    # Each bump: exp(-scale * ((t - t_c)^2 + (x - x_c)^2 + (v - v_c)^2))
    
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
    
    dt3 = t - t_c3
    dx3 = x - x_c3
    dv3 = v - v_c3
    r2_3 = dt3**2 + dx3**2 + dv3**2
    g3 = np.exp(-scale * r2_3)
    
    # Weighted sum with fixed positive weights (the weights are implicitly 1,
    # but we can let the optimizer adjust via the centers and scale)
    # To allow more flexibility, we use the parameters as centers and scale,
    # and the output is a weighted combination. Since we only have 10 params,
    # we use 3 bumps with shared scale and unit weights.
    # Actually, let's use the remaining structure: the 10 params are fully used
    # as 3 centers (9 params) + 1 scale. The weights are fixed at 1.0.
    # But this limits expressiveness. Let's instead interpret differently:
    # Use 2 bumps with full flexibility: 2*(center_t, center_x, center_v, weight) + 2 scale params?
    # That's 2*4 + 2 = 10. Let's do that.
    
    # Reinterpret:
    # params[0:4] = (t_c1, x_c1, v_c1, w1)
    # params[4:8] = (t_c2, x_c2, v_c2, w2)
    # params[8]   = scale1
    # params[9]   = scale2
    
    t_c1, x_c1, v_c1, w1 = params[0], params[1], params[2], params[3]
    t_c2, x_c2, v_c2, w2 = params[4], params[5], params[6], params[7]
    scale1 = np.maximum(np.abs(params[8]), 1e-3)
    scale2 = np.maximum(np.abs(params[9]), 1e-3)
    
    # Recompute with 2 bumps
    dt1 = t - t_c1
    dx1 = x - x_c1
    dv1 = v - v_c1
    r2_1 = dt1**2 + dx1**2 + dv1**2
    g1 = np.exp(-scale1 * r2_1)
    
    dt2 = t - t_c2
    dx2 = x - x_c2
    dv2 = v - v_c2
    r2_2 = dt2**2 + dx2**2 + dv2**2
    g2 = np.exp(-scale2 * r2_2)
    
    # Add a linear baseline term to help capture the overall trend
    # But we've used all 10 params. Let's include a small constant offset
    # derived from the weights to ensure the model can represent a constant.
    
    return w1 * g1 + w2 * g2
# EVOLVE-BLOCK-END
