"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid rational model: Linear polynomial + 2 Gaussian Radial Basis Functions.
    
    y(x) = p0 + p1*x + sum_{i=1}^{2} w_i * exp(-(x - c_i)^2 / sigma^2)
    
    This model combines a global linear trend with local Gaussian bumps.
    The Gaussians allow capturing non-linear features, local extrema, or
    sharp transitions that a pure polynomial might miss, while the linear
    part handles the overall trend. The parameters are linear in the basis
    functions, which helps BFGS convergence, although the centers c_i
    introduce non-linearity.
    
    Parameters (10 total):
    params[0]: p0 (intercept)
    params[1]: p1 (slope)
    params[2]: w1 (amplitude of Gaussian 1)
    params[3]: c1 (center of Gaussian 1)
    params[4]: w2 (amplitude of Gaussian 2)
    params[5]: c2 (center of Gaussian 2)
    params[6..9]: Unused (set to 0 to avoid side effects, though they are part of the array)
    
    Note: The task specifies params is length 10. We use only 6.
    The remaining 4 are ignored.
    """
    # Ensure x is a 1D array of floats
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    
    w1 = params[2]
    c1 = params[3]
    
    w2 = params[4]
    c2 = params[5]
    
    # Linear part
    linear_part = p0 + p1 * x
    
    # Gaussian part
    # We choose a fixed width sigma^2 = 0.5 (sigma ~ 0.707)
    # This provides a balance between local resolution and global coverage in [-2, 2].
    # exp(-(x-c)^2 / 0.5) = exp(-2*(x-c)^2)
    
    diff1 = x - c1
    diff2 = x - c2
    
    # Clip the exponent argument to avoid overflow/underflow issues, 
    # although for negative arguments exp is safe. 
    # However, if c1 or c2 go crazy, diff can be huge.
    # exp(-huge) -> 0, which is fine.
    # exp(huge) -> inf, which is bad. But the argument is always negative or zero.
    # So we are safe from overflow. Underflow to 0 is fine.
    
    g1 = np.exp(-2.0 * diff1 * diff1)
    g2 = np.exp(-2.0 * diff2 * diff2)
    
    gaussian_part = w1 * g1 + w2 * g2
    
    result = linear_part + gaussian_part
    
    # Safety check for non-finite values
    # If parameters diverge, result might be inf/nan.
    # Replace non-finite with a large value to signal high error to optimizer,
    # or 0 if we want to be benign? 
    # The contract says non-finite error costs 1e30. 
    # If we return inf, MSE is inf. 
    # If we return 0, MSE is finite but likely wrong.
    # Returning inf/nan is usually better for the optimizer to step back.
    # However, some optimizers handle nan poorly. 
    # Let's replace nan with a large finite number to ensure the loss function
    # returns a finite (but huge) value, which might help BFGS step back.
    # Actually, the contract says "non-finite error costs 1e30 during fitting".
    # This implies the evaluator itself might handle non-finite outputs by assigning 1e30.
    # So we can just return the raw result.
    
    return result
# EVOLVE-BLOCK-END
