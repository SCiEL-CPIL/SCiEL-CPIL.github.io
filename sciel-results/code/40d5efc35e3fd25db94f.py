"""Symbolic regression with Gaussian Process (Matérn-5/2 kernel) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    GP-based model using a Matérn-5/2 kernel.
    
    The 10 parameters are arranged as:
    - params[0]: signal variance (sigma_f^2)
    - params[1]: length scale for x (l_x)
    - params[2]: length scale for y (l_y)
    - params[3]: length scale for z (l_z)
    - params[4]: mean offset (mu)
    - params[5-9]: additional basis coefficients for a small polynomial expansion
      to capture residual structure not captured by the GP mean.
    
    We use a fixed set of training points (the inputs themselves serve as the
    'support points' since we have no separate training set visible here).
    Actually, since we don't have access to the training targets, we must use
    a parametric form. Let's use a combination of:
    - A constant term
    - Linear terms in x, y, z
    - Quadratic terms in x, y, z
    - Pairwise interaction terms
    But that's 1+3+3+3=10 terms. However, the directive suggests GP.
    
    Since we cannot fit a GP without knowing the training targets (the evaluator
    fits params to minimize training MSE, but we don't have the targets), we
    must use a parametric model. The GP idea would require the targets.
    
    Instead, let's use a richer polynomial basis that might capture more structure.
    We'll use:
    - params[0]: constant
    - params[1]: x
    - params[2]: y
    - params[3]: z
    - params[4]: x^2
    - params[5]: y^2
    - params[6]: z^2
    - params[7]: x*y
    - params[8]: x*z
    - params[9]: y*z
    
    This is a full quadratic polynomial in 3 variables.
    """
    p = params
    
    # Full quadratic polynomial basis
    result = (p[0] * 1.0
              + p[1] * x
              + p[2] * y
              + p[3] * z
              + p[4] * x * x
              + p[5] * y * y
              + p[6] * z * z
              + p[7] * x * y
              + p[8] * x * z
              + p[9] * y * z)
    
    return result
# EVOLVE-BLOCK-END
