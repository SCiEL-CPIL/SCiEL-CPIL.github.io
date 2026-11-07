# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # Strategy: Radial Basis Function (RBF) Network
    # Model: y(x) = sum_{j=1}^{9} c_j * exp(-(x - center_j)^2 / (2 * sigma^2)) + c_0
    # 
    # Parameters mapping:
    # p[0]: Constant offset (bias)
    # p[1]: Logarithm of the kernel width sigma. 
    #       Using log ensures sigma > 0 during optimization and helps BFGS 
    #       navigate the scale of the width parameter more effectively.
    # p[2..10]: Coefficients for the 9 RBF kernels.
    
    # Fixed centers evenly spaced in [-3, 3]
    # We use 9 centers.
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Extract parameters
    offset = p[0]
    log_sigma = p[1]
    coeffs = p[2:11] # 9 coefficients
    
    # Calculate sigma
    # Clip log_sigma to avoid extreme values that might cause under/overflow
    # exp(10) ~ 22000, exp(-10) ~ 4.5e-5. 
    # If sigma is too large, RBFs become nearly constant (redundant with offset).
    # If sigma is too small, RBFs become spikes (numerically unstable/overfitting).
    # Let's constrain log_sigma to [-5, 5] effectively, but soft clipping is better for gradients.
    # However, BFGS handles bounds poorly without constraints. 
    # We can use a sigmoid-like scaling or just let it explore, but prevent overflow.
    # np.exp(log_sigma) is safe for log_sigma < 700.
    # To keep the optimization landscape well-behaved, we can map p[1] through a tanh or similar?
    # No, direct exp is standard for log-parameters. 
    # Let's just ensure it doesn't become NaN.
    
    sigma = np.exp(log_sigma)
    
    # If sigma is extremely small or large, the result might be degenerate.
    # But BFGS should find a reasonable sigma if the function is smooth.
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent term
    # To avoid overflow in exp if sigma is very small and diff is large:
    # -diff^2 / (2 * sigma^2)
    # If sigma -> 0, this term -> -inf (for diff != 0), exp -> 0. Correct.
    # If sigma -> inf, this term -> 0, exp -> 1. Correct.
    
    # Check for division by zero or non-finite sigma
    if not np.isfinite(sigma) or sigma <= 0:
        # Fallback to linear if sigma is invalid
        return p[0] + p[1] * x
        
    exponent = -0.5 * (diff ** 2) / (sigma ** 2)
    
    # Clip exponent to prevent underflow to 0 too aggressively? 
    # Underflow to 0 is mathematically correct for large distances/small sigma.
    # Overflow in exp? exponent is always <= 0. So exp(exponent) is in (0, 1]. No overflow.
    
    rbf_vals = np.exp(exponent)
    
    # Compute linear combination
    # result = offset + sum(coeffs * rbf_vals)
    result = offset + np.sum(rbf_vals * coeffs[np.newaxis, :], axis=1)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback to simple linear combination
        return p[0] + p[1] * x
        
    return result
# EVOLVE-BLOCK-END
