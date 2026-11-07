"""
Strategy: Radial Basis Function (RBF) Approximator.
Instead of a global polynomial basis which suffers from collinearity and limited 
expressiveness for complex nonlinearities, we use 3 Gaussian RBFs.
The 10 parameters are allocated as:
- 3 parameters for the center of RBF 1 (tx, tx, tv)
- 3 parameters for the center of RBF 2 (cx2, cx2, cv2)
- 3 parameters for the center of RBF 3 (cx3, cx3, cv3)
- 1 parameter for the global scale (width) of the Gaussians.
Wait, we need output weights too. 
Let's adjust:
We have 10 params.
Let's use 2 RBFs to leave room for weights and scale.
Params:
p0, p1, p2: Center of RBF 1 (t, x, v)
p3, p4, p5: Center of RBF 2 (t, x, v)
p6: Width (sigma) of the Gaussians (shared for simplicity and stability)
p7: Weight for RBF 1
p8: Weight for RBF 2
p9: Bias / Global offset

This allows the function to be:
y = p9 + p7 * exp(-||[t,x,v] - C1||^2 / (2*p6^2)) + p8 * exp(-||[t,x,v] - C2||^2 / (2*p6^2))

To prevent division by zero if p6 is close to 0 during BFGS, we add a small epsilon.
Also, to handle the scale of inputs (t in [30,50], x in [-0.2,0.17], v in [-0.3,0.36]),
the centers will be fitted to these ranges.
"""
import numpy as np
import time

# Deadline guard to ensure we finish within limits if called in a loop (though this is a single call)
# The evaluator calls equation() once per fit iteration. This function itself must be fast.
# No internal search is performed here; the "search" is the BFGS fitting done by the evaluator.
# We just need to provide a differentiable, flexible function.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a 2-Gaussian RBF basis.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 floats
    
    Returns:
    prediction: array of shape (N,)
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Centers
    c1_t, c1_x, c1_v = p0, p1, p2
    c2_t, c2_x, c2_v = p3, p4, p5
    
    # Width (sigma). Ensure it's not too small to cause numerical instability in exp
    sigma = p6
    # If sigma is close to 0, the Gaussian becomes a spike. 
    # We clamp sigma to a small positive value to maintain differentiability and finiteness.
    # However, BFGS might try to push sigma to 0. 
    # Using a softplus-like or just abs + eps is safer.
    # Let's use a lower bound of 1e-3 for sigma.
    sigma_safe = np.maximum(np.abs(sigma), 1e-3)
    
    # Precompute squared distances
    # dist1^2 = (t - c1_t)^2 + (x - c1_x)^2 + (v - c1_v)^2
    dt1 = t - c1_t
    dx1 = x - c1_x
    dv1 = v - c1_v
    dist1_sq = dt1**2 + dx1**2 + dv1**2
    
    # dist2^2 = (t - c2_t)^2 + (x - c2_x)^2 + (v - c2_v)^2
    dt2 = t - c2_t
    dx2 = x - c2_x
    dv2 = v - c2_v
    dist2_sq = dt2**2 + dx2**2 + dv2**2
    
    # Gaussian basis functions
    # exp(-dist^2 / (2 * sigma^2))
    # To avoid overflow/underflow issues, we rely on numpy's exp which handles large negative numbers well.
    # The denominator is 2 * sigma_safe^2.
    denom = 2.0 * sigma_safe**2
    
    rbf1 = np.exp(-dist1_sq / denom)
    rbf2 = np.exp(-dist2_sq / denom)
    
    # Combine with weights and bias
    # p7 is weight for rbf1, p8 is weight for rbf2, p9 is bias
    result = p9 + p7 * rbf1 + p8 * rbf2
    
    return result
