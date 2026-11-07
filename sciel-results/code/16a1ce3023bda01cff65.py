import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed-center Gaussian RBF network.
    
    The function is modeled as a sum of Gaussian bumps plus a constant:
    y(x) = c + sum_i v_i * exp(-((x - c_i)^2) / (2 * w_i^2))
    
    Strategy:
    - Use 9 Gaussian units.
    - Fix centers c_i evenly spaced across the domain [-2, 2].
    - Fix widths w_i to a reasonable value (e.g., 0.4) to ensure local sensitivity 
      but sufficient overlap for smooth combination.
    - The 10 parameters are the 9 amplitudes v_i and the constant offset c.
    
    This design makes the optimization landscape for BFGS nearly convex with 
    respect to the parameters (v_i, c), as the basis functions are fixed. 
    This avoids the local minima traps of learning both weights and biases 
    simultaneously, which plagued the tanh network approach.
    """
    # Extract params: 9 amplitudes and 1 constant
    v = params[:9]
    c = params[9]
    
    # Fixed centers: 9 points evenly spaced in [-2, 2]
    # np.linspace(-2, 2, 9) gives [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    centers = np.linspace(-2.0, 2.0, 9)
    
    # Fixed width: 0.4 provides a good balance between locality and smoothness
    # sigma = 0.4 -> sigma^2 = 0.16 -> 2*sigma^2 = 0.32
    sigma_sq = 0.16
    two_sigma_sq = 2.0 * sigma_sq
    
    # Compute Gaussian basis functions
    # x shape: (N,)
    # centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # exp(-diff^2 / (2*sigma^2))
    # Use clipping to prevent underflow/overflow issues in exp, though Gaussians are safe
    # but large negative exponents are fine.
    gaussians = np.exp(-diff**2 / two_sigma_sq)
    
    # Combine: y = v @ gaussians + c
    # v shape: (9,)
    # gaussians shape: (N, 9)
    # v @ gaussians.T -> (N,)
    y = gaussians @ v + c
    
    return y
