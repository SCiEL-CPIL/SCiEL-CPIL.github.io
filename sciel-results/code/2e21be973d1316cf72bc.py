"""
Alternative strategy: Gaussian Mixture Model basis.
Instead of a global polynomial expansion, we use 5 Gaussian 'bumps' with
learnable centers, widths, and amplitudes. This allows the BFGS optimizer
to locally adapt to the shape of the hidden function, which can be more
effective than trying to fit a single global polynomial with 10 coefficients.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a sum of 5 Gaussian functions.
    params layout (length 10):
    params[0:5]  -> Amplitudes (a_i)
    params[5:10] -> Centers (mu_i)
    
    Note: Widths (sigma_i) are fixed to 0.5 to reduce the non-convexity 
    of the optimization landscape, allowing BFGS to focus on positioning 
    and scaling the Gaussians.
    """
    # Extract parameters
    # We have 10 params. Let's use 5 Gaussians.
    # 5 amplitudes, 5 centers. Widths are fixed.
    # This leaves the problem with 10 free variables, matching the requirement.
    
    amplitudes = params[0:5]
    centers = params[5:10]
    
    # Fixed widths to prevent collapse or explosion
    # 0.5 is a reasonable width for x in [-2, 2] to cover ~1 unit range per bump
    sigma = 0.5
    
    # Compute Gaussian terms
    # x shape: (N,)
    # centers shape: (5,)
    # We want to compute sum_i a_i * exp(-0.5 * ((x - mu_i) / sigma)^2)
    
    # Efficient vectorized computation
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exponent
    # exp_term shape: (N, 5)
    exp_term = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum
    # amplitudes shape: (5,) -> (1, 5) for broadcasting
    result = exp_term @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
