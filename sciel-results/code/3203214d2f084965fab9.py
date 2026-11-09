"""
Strategy: Adaptive Non-Linear RBF Basis Search.

Unlike the current program's fixed uniform RBF grid (linear in params), this program
uses the 10 parameters to define the basis functions themselves:
- params[0:8]: 8 RBF centers (c_i) and widths (w_i) for 4 RBFs? 
  Wait, we have 10 params.
  Let's use:
  - params[0:5]: Centers c_1..c_5
  - params[5:9]: Widths w_1..w_4 (4 widths? No, let's keep it simple)
  
  Better allocation for 10 params:
  - 5 RBFs.
  - params[0:5]: Centers c_1..c_5
  - params[5:9]: Widths w_1..w_4? No, that's asymmetric.
  
  Let's try:
  - 4 RBFs + 1 Linear Term + 1 Offset?
  - Or 5 RBFs where widths are derived from centers?
  
  Actually, a common effective approach is to use a small number of RBFs with free centers and widths.
  With 10 params:
  - 5 Centers (c1..c5)
  - 5 Widths (w1..w5)
  - But then we have 0 params for coefficients?
  
  The model must be linear in params for BFGS to be robust? No, BFGS can handle non-linear.
  But if the model is y = sum( a_i * exp(-(x-c_i)^2 / w_i^2) ), we need coefficients a_i too.
  That would require 15 params. We only have 10.
  
  Alternative: Fix the coefficients to be 1 (or derived) and only optimize centers/widths?
  That's very restrictive.
  
  Let's look at the constraint: "The evaluator fits params... minimizing training MSE."
  The evaluator calls `equation(x, params)`.
  
  If I make the model non-linear in params, BFGS will try to optimize the non-linear part.
  This is a harder optimization problem.
  
  However, the current program (Score 340) uses a linear RBF model.
  Inspiration 1 (Score 4384) uses Chebyshev (linear).
  Inspiration 3 (Score 157) is the current program's strategy (RBF).
  
  The knowledge base says: "Chebyshev basis offers superior conditioning... allowing BFGS to converge... Measured: 157.3 -> 4385."
  This implies that for smooth functions, linear bases with good conditioning (Chebyshev) are better than RBFs or other linear bases.
  
  BUT, the task is to find a DIFFERENT strategy.
  
  What if the function is NOT smooth? Or has specific local features?
  A linear RBF with fixed centers might miss a sharp peak if the center isn't there.
  A non-linear RBF (free centers) could move to the peak.
  
  Let's try a hybrid:
  Use 4 RBFs.
  Params:
  0-3: Centers c1..c4
  4-7: Widths w1..w4
  8:   Global Scale (multiplies the sum of RBFs)
  9:   Offset (bias)
  
  Model: y = scale * sum( exp(-(x - c_i)^2 / (2 * w_i^2)) ) + offset
  Note: Coefficients of RBFs are fixed to 1. This is a strong constraint, but it reduces the dimensionality of the non-linear part.
  
  Why this might work better:
  If the target function is a sum of Gaussians with unknown centers/widths but similar amplitudes, this model is perfect.
  If the target is a polynomial, this model will struggle, but BFGS might push centers to the edges or widths to infinity (approximating constants/linear terms poorly).
  
  Risk: BFGS starting from all ones might get stuck in local minima because the landscape is non-convex.
  Mitigation: The problem is 1D, so the landscape is simpler than higher dimensions.
  
  Let's implement this.
  
  Parameters:
  c1, c2, c3, c4: Centers
  w1, w2, w3, w4: Widths (must be > 0)
  s: Scale
  b: Bias
  
  To ensure widths are positive, we can use softplus or just square them?
  If w is a param, w^2 is always positive.
  Let's use w_i = param_i. Then use exp(-(x-c)^2 / (2 * w^2)).
  If w is close to 0, the RBF becomes a spike.
  If w is large, it becomes a broad bump.
  
  Starting from all ones:
  Centers at 1.0. Widths at 1.0. Scale 1.0. Bias 1.0.
  This is a valid starting point.
  
  One issue: Symmetry. If the function is symmetric, BFGS might struggle with degenerate centers.
  But with 4 centers, it has flexibility.
  
  Let's try this.
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes prediction using 4 adaptive RBFs with free centers and widths.
    
    Params:
    0-3: Centers c1..c4
    4-7: Widths w1..w4 (squared in denominator)
    8:   Scale
    9:   Bias
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Extract parameters
    centers = params[0:4]
    widths = params[4:8]
    scale = params[8]
    bias = params[9]
    
    # Avoid division by zero or extremely small widths causing overflow/underflow issues
    # Clip widths to a small positive value
    widths = np.clip(widths, 1e-6, None)
    
    y = np.zeros_like(x)
    
    for i in range(4):
        c = centers[i]
        w = widths[i]
        # Gaussian kernel
        # exp(-(x - c)^2 / (2 * w^2))
        # We want to ensure numerical stability.
        # If w is very small, (x-c)^2 / w^2 can be huge. exp(-huge) = 0. This is fine.
        # If w is huge, it's a broad bump.
        
        dist_sq = (x - c) ** 2
        y += np.exp(-dist_sq / (2.0 * w * w))
        
    y = scale * y + bias
    
    return y
