"""
Strategy: Locally Adaptive Gaussian Basis (Neural Network-like)
Instead of a fixed global basis, we use 5 Gaussian bumps with learnable centers and widths.
This allows the model to fit local features more accurately than a fixed polynomial/trig basis.
The parameters are:
- p[0:5]: Amplitudes of the 5 Gaussians
- p[5:9]: Centers of the first 4 Gaussians (the 5th is fixed at 0 to ensure central coverage)
- p[9]: Width scale (shared for all Gaussians to maintain stability and reduce parameters)
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 5 Gaussian basis functions
    with learnable centers and a shared width.
    """
    # Ensure x is a 1D float array
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    
    # Parameters
    # p[0:5] are amplitudes
    # p[5:9] are centers for the first 4 Gaussians
    # p[9] is the shared width parameter (inverse of sigma squared, effectively)
    
    # Extract parameters
    amps = p[0:5]
    centers = p[5:9]
    # Fixed center for the 5th Gaussian at 0
    centers = np.append(centers, 0.0)
    
    # Shared width: use p[9] as a scale factor. 
    # To ensure stability and avoid division by zero or extreme values, 
    # we map p[9] to a reasonable range or use it directly as 1/sigma^2.
    # Let's use exp(p[9]) to ensure positive width, but BFGS might struggle with exp.
    # Better: use p[9] directly as a variance-like parameter, but ensure it's not too small.
    # Or simply use a fixed reasonable width if p[9] is unstable? 
    # The inspiration suggests learnable centers and widths.
    # Let's use p[9] as the inverse of the squared width (sigma^2).
    # To avoid numerical issues if p[9] is negative or zero, we can clip it or add a small epsilon.
    # However, BFGS works with real numbers. A common trick is to use the parameter directly 
    # in the exponent with a sign check or just assume the optimizer finds a positive value.
    # Given the starting point is all ones, p[9]=1, which is a reasonable width (sigma^2=1).
    
    inv_width_sq = p[9]
    # If inv_width_sq is very small or negative, the Gaussians become very wide or oscillatory (if negative).
    # To be safe and robust, we can clip it to a minimum positive value.
    inv_width_sq = np.maximum(inv_width_sq, 0.1)
    
    # Calculate Gaussians
    # centers shape: (5,), x shape: (N,)
    # We want a matrix of shape (5, N) to compute exp(-0.5 * inv_width_sq * (x - centers)^2)
    # Then multiply by amps (5,) and sum.
    
    # Efficient computation:
    # diff = x[None, :] - centers[:, None]  -> shape (5, N)
    # gaussians = np.exp(-0.5 * inv_width_sq * diff**2)
    # y = amps @ gaussians
    
    diff = x[None, :] - centers[:, None]
    gaussians = np.exp(-0.5 * inv_width_sq * diff**2)
    y = np.dot(amps, gaussians)
    
    return y

# EVOLVE-BLOCK-END
