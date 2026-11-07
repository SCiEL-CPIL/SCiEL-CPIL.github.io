"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 5 centers and 5 widths.
    params[0:5] are the centers c_i
    params[5:10] are the widths sigma_i (absolute value used to ensure positivity)
    
    The output is a weighted sum of Gaussian kernels:
    y = sum_i exp(-(x - c_i)^2 / sigma_i^2)
    
    Note: The weights are fixed to 1. This allows the model to fit functions
    that are sums of Gaussian bumps. To allow negative contributions, we could
    use alternating signs or let the optimizer find negative widths? No, width must be positive.
    Actually, if all weights are 1, we can only model positive bumps.
    A better strategy: Use 5 terms, but allow the 'center' to effectively scale? No.
    
    Let's refine: The prompt says 10 params.
    If I use 5 centers and 5 widths, I have no free weights for the amplitude.
    This restricts the model to sum of unit-amplitude Gaussians.
    This might be too restrictive for arbitrary functions (e.g., y = 0.1 * exp(...)).
    
    Alternative: 5 centers, 5 weights? But then widths are fixed?
    If widths are fixed, the model is a linear combination of 5 fixed RBFs.
    This is a standard RBF regression setup.
    Fixed widths: e.g., [0.5, 1.0, 2.0, 0.2, 1.5] or similar.
    Centers are learnable.
    Weights are learnable.
    Total params: 5 centers + 5 weights = 10.
    
    Let's try this:
    params[0:5] = centers c_i
    params[5:10] = weights w_i
    Fixed widths sigma_i = [0.3, 0.7, 1.0, 1.5, 2.0]
    
    y = sum_i w_i * exp(-(x - c_i)^2 / sigma_i^2)
    
    This is a universal approximator (with enough centers). With 5 centers, it's good for smooth functions.
    BFGS should be able to optimize centers and weights.
    Starting from all ones:
    Centers start at 1.0.
    Weights start at 1.0.
    
    Potential issue: If all centers start at 1.0, they are identical.
    Gradient might be symmetric, leading to slow convergence or stuck local minima.
    To break symmetry, we can add a small constant offset to the initial centers in the function?
    No, the function receives params from the optimizer. The optimizer starts at 1.0.
    We can't change the initial values inside equation().
    
    However, if centers are identical, the gradients with respect to centers might be zero or symmetric.
    Let's check: d/dc exp(-(x-c)^2/sigma^2) = 2(x-c)/sigma^2 * exp(...).
    If c=1 for all, and x varies, the gradients are different for different x.
    So the optimizer will likely move them apart.
    
    Let's implement this.
    """
    centers = params[0:5]
    weights = params[5:10]
    
    # Fixed widths to provide different scales
    sigmas = np.array([0.3, 0.7, 1.0, 1.5, 2.0])
    
    # Compute RBF terms
    # x shape: (N,)
    # centers shape: (5,)
    # We need (N, 5) differences
    x_reshaped = x[:, np.newaxis]
    centers_reshaped = centers[np.newaxis, :]
    
    diffs = x_reshaped - centers_reshaped
    # sigmas shape: (5,)
    sigmas_reshaped = sigmas[np.newaxis, :]
    
    # Gaussian kernel
    # exp(-diffs^2 / sigmas^2)
    kernel_values = np.exp(-(diffs ** 2) / (sigmas_reshaped ** 2))
    
    # Weighted sum
    # kernel_values shape: (N, 5)
    # weights shape: (5,)
    y = np.sum(kernel_values * weights[np.newaxis, :], axis=1)
    
    return y
# EVOLVE-BLOCK-END
