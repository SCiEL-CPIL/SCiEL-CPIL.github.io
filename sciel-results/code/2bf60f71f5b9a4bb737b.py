"""RBF-based symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 10 Gaussian kernels.
    params[0] = inverse width (beta) for all kernels
    params[1:10] = weights for 10 fixed centers
    
    Centers are fixed at evenly spaced points in [0.1, 6].
    """
    # Fixed centers spanning [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Learnable inverse width; ensure it stays positive and reasonable
    beta = params[0]
    # Use softplus-like transformation to keep beta positive and avoid extreme values
    # Actually, let's just use beta directly but clip to avoid numerical issues
    # BFGS starts at 1.0, so beta will be near 1 initially
    # We need to be careful: if beta is negative, exp(-beta * dist^2) grows
    # Let's use abs or a soft transformation. Actually, for RBF we want beta > 0.
    # Let's use beta = 1.0 / (1.0 + np.exp(-params[0])) * 5.0 + 0.1
    # This maps any real number to (0.1, 5.1), a reasonable width range.
    # But wait, BFGS optimizes in the original parameter space. If we transform,
    # the gradient is with respect to the transformed parameter. That's fine.
    # Actually, simpler: just use params[0] as the width parameter directly,
    # but ensure numerical stability.
    
    # Let's use a safer approach: width = 1.0 / (1.0 + np.exp(-params[0]))
    # This gives width in (0, 1), which means the Gaussian is quite narrow.
    # Or width = 2.0 + np.exp(params[0]) to keep it > 2.
    # Let's think about what's reasonable. The domain is [0.1, 6], span ~5.9.
    # With 10 centers, spacing is ~0.65. A good width might be ~0.5-1.0.
    # Let's use: sigma = 0.5 + 0.5 * np.exp(params[0])  # sigma in (0.5, inf)
    # Or: sigma = 1.0 / (1.0 + np.exp(-params[0])) * 3.0 + 0.2  # sigma in (0.2, 3.2)
    
    # Let's go with: sigma = 0.3 + 1.0 / (1.0 + np.exp(-params[0])) * 2.0
    # This gives sigma in (0.3, 2.3), which is reasonable.
    sigma = 0.3 + 1.0 / (1.0 + np.exp(-params[0])) * 2.0
    
    # Compute distances from each x to each center
    # x shape: (n_points,)
    # centers shape: (10,)
    # dist shape: (n_points, 10)
    dist = np.abs(x[:, np.newaxis] - centers[np.newaxis, :])
    
    # Gaussian kernels: exp(-dist^2 / (2 * sigma^2))
    # Use 2*sigma^2 in denominator for standard Gaussian
    kernel = np.exp(-dist**2 / (2.0 * sigma**2))
    
    # Weights are params[1:10]
    weights = params[1:10]
    
    # Prediction: sum of weight_i * kernel(x, c_i)
    prediction = kernel @ weights
    
    return prediction
# EVOLVE-BLOCK-END
