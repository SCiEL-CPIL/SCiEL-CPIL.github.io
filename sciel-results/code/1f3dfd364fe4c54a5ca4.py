"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 4 Gaussian bumps + 4 polynomial terms + 2 linear terms.
    This provides local flexibility (Gaussians) for non-smooth features
    and global stability (polynomials) for smooth trends.
    """
    # Gaussian basis functions centered at strategic points in [-2, 2]
    # Centers: -1.5, -0.5, 0.5, 1.5 with width ~0.8
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    width = 0.8
    
    # Compute Gaussian terms: exp(-(x - c)^2 / (2*width^2))
    x_expanded = x[:, np.newaxis]
    gaussians = np.exp(-((x_expanded - centers) ** 2) / (2 * width ** 2))
    # gaussians shape: (n_points, 4)
    
    # Polynomial terms: x^0, x^1, x^2, x^3
    poly = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3
    ])
    # poly shape: (n_points, 4)
    
    # Linear interaction terms for additional flexibility
    linear_terms = np.column_stack([
        x * np.sin(np.pi * x / 4.0),  # bounded oscillatory component
        x * np.cos(np.pi * x / 2.0)   # another bounded oscillatory component
    ])
    # linear_terms shape: (n_points, 2)
    
    # Combine all basis terms: 4 gaussians + 4 polys + 2 linear = 10 total
    # params[0:4] for gaussians, params[4:8] for polynomials, params[8:10] for linear
    
    result = (gaussians[:, 0] * params[0] +
              gaussians[:, 1] * params[1] +
              gaussians[:, 2] * params[2] +
              gaussians[:, 3] * params[3] +
              poly[:, 0] * params[4] +
              poly[:, 1] * params[5] +
              poly[:, 2] * params[6] +
              poly[:, 3] * params[7] +
              linear_terms[:, 0] * params[8] +
              linear_terms[:, 1] * params[9])
    
    return result
# EVOLVE-BLOCK-END
