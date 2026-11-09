"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < TIME_LIMIT


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model where the 10 parameters define the function structure.
    
    Parameters are interpreted as:
      p[0]: amplitude scale (positive via softplus)
      p[1]: frequency for x-component
      p[2]: frequency for y-component
      p[3]: frequency for interaction x*y
      p[4]: power/exponent for x
      p[5]: power/exponent for y
      p[6]: weight for sin component
      p[7]: weight for cos component
      p[8]: bias/shift
      p[9]: secondary amplitude for higher-order terms
    
    The model combines:
      - Trigonometric terms with learnable frequencies
      - Polynomial-like terms with learnable exponents (via exp/log for smoothness)
      - Interaction terms
      - A global scale and shift
    
    This gives BFGS a smooth, nonlinear optimization landscape where each parameter
    has a clear semantic role, which should converge better than a linear feature model
    from the all-ones initialization.
    """
    n = len(x)
    
    # Extract parameters with safe transformations
    # Use softplus-like transformation for positive quantities
    # p[0]: amplitude (always positive)
    amp = np.exp(np.clip(params[0], -10, 10))  # positive scale
    
    # p[1], p[2], p[3]: frequencies (can be negative, but bounded)
    fx = np.tanh(params[1]) * 5.0  # frequency for x, range [-5, 5]
    fy = np.tanh(params[2]) * 5.0  # frequency for y, range [-5, 5]
    fxy = np.tanh(params[3]) * 5.0  # frequency for interaction, range [-5, 5]
    
    # p[4], p[5]: exponents (bounded to avoid extreme values)
    ex = 1.0 + np.tanh(params[4])  # exponent for x, range [0, 2]
    ey = 1.0 + np.tanh(params[5])  # exponent for y, range [0, 2]
    
    # p[6], p[7]: weights for sin/cos components
    w_sin = np.tanh(params[6])  # weight for sine terms, [-1, 1]
    w_cos = np.tanh(params[7])  # weight for cosine terms, [-1, 1]
    
    # p[8]: bias
    bias = params[8]
    
    # p[9]: secondary amplitude for higher-order terms
    amp2 = np.exp(np.clip(params[9], -10, 10))  # positive secondary scale
    
    # Build the nonlinear model
    # Component 1: Sine terms with learnable frequencies
    sin_x = np.sin(fx * x)
    sin_y = np.sin(fy * y)
    sin_xy = np.sin(fxy * x * y)
    
    # Component 2: Cosine terms
    cos_x = np.cos(fx * x)
    cos_y = np.cos(fy * y)
    
    # Component 3: Polynomial-like terms with learnable exponents
    # Use sign-preserving power: sign(x)*|x|^ex
    x_pow = np.sign(x) * np.abs(x) ** ex
    y_pow = np.sign(y) * np.abs(y) ** ey
    
    # Component 4: Interaction term
    xy_term = x * y
    
    # Combine all components
    pred = (amp * (w_sin * (sin_x + sin_y + sin_xy) + 
                   w_cos * (cos_x + cos_y) + 
                   1.0 * (x_pow + y_pow) + 
                   0.5 * xy_term) +
            amp2 * (x ** 2 + y ** 2) +
            bias)
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
