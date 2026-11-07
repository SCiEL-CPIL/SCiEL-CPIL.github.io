"""Bootstrap portfolio: Adaptive RBF model with learnable centers and widths."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Radial Basis Function (RBF) model.
    
    The function is approximated as a weighted sum of Gaussian bumps.
    We use 10 parameters: 5 for weights, 5 for widths (log-parameterized).
    Centers are fixed at a grid of 5 points (but we have 10 params, so
    we use 5 RBFs with learnable widths, or we can use 10 RBFs with fixed widths).
    
    Let's try: 10 RBFs with fixed centers (a 3x3 grid + 1 center) but only
    10 params for weights. That's what the current code does.
    
    Alternative: Use 5 RBFs with 5 weights and 5 widths (log-scale).
    Centers: 5 fixed points.
    
    Or: 10 RBFs with 10 weights, but also make the widths adaptive?
    We only have 10 params total.
    
    Let's try a different structural change:
    Use a combination of polynomial terms AND RBF terms.
    
    Actually, let's look at the inspiration: "Gaussian Process (RBF) kernel basis
    with fixed centers distributed across the domain".
    
    The current program already does this. Why did it score 2.89?
    Maybe the centers are not well distributed, or the width is not optimal.
    
    Let's try a different approach:
    Use a **tensor product** of 1D RBFs.
    f(x, y) = sum_i sum_j w_ij * phi(x - xi) * phi(y - yj)
    
    With 10 params, we can have a 3x3 grid of centers (9 params) + 1 global scale?
    Or a 2x5 grid?
    
    Let's try a 3x3 grid of centers for x and y, with 9 weights, and 1 parameter
    for the width (shared across all RBFs).
    
    Centers for x: [-1, 0, 1]
    Centers for y: [-1, 0, 1]
    
    This gives 9 RBFs. The 10th parameter can be a constant offset.
    
    This is a structured RBF network that should be able to capture
    separable functions well, and also non-separable ones to some extent.
    """
    p = params
    
    # 3x3 grid of centers
    x_centers = np.array([-1.0, 0.0, 1.0])
    y_centers = np.array([-1.0, 0.0, 1.0])
    
    # Width parameter (index 9), use softplus or exp to ensure positive
    # log-width for better conditioning
    log_sigma = p[9]
    sigma = np.exp(log_sigma)
    sigma_sq_2 = 2.0 * sigma * sigma
    
    result = np.zeros_like(x)
    
    idx = 0
    for cx in x_centers:
        for cy in y_centers:
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            kernel = np.exp(-dist_sq / sigma_sq_2)
            result += p[idx] * kernel
            idx += 1
    
    return result
# EVOLVE-BLOCK-END
