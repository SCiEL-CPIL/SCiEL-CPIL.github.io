"""
Strategy: Local Gaussian Radial Basis Functions (RBFs).

This strategy replaces the global Chebyshev polynomial basis with a local basis of
Gaussian Radial Basis Functions.

Rationale:
1. Local Flexibility: Unlike global polynomials (Chebyshev), RBFs allow the model to
   capture local features, sharp variations, or non-uniform behavior in the function
   without being constrained by a global smoothness assumption. If the hidden function
   has local "bumps" or varies significantly in specific regions, RBFs can adapt by
   weighting specific centers.
2. Handling Non-Polynomial Shapes: If the target function is not well-approximated by
   a low-degree polynomial (e.g., it has discontinuities, sharp peaks, or high-frequency
   components), Chebyshev series will suffer from Gibbs-like phenomena or poor fit.
   RBFs can model such shapes more effectively with the same number of parameters.
3. Linear in Parameters: The model remains linear in the weights (params), ensuring
   that the BFGS optimizer converges to the global least-squares solution for the
   fixed centers and widths.
4. Fixed Hyperparameters: To keep the problem within the 10-parameter constraint, we
   fix the centers to be uniformly distributed over [-3, 3] and fix the width (sigma)
   to 1.0. This provides a reasonable balance between local resolution and smoothness.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the Gaussian RBF expansion.
    
    Args:
        x: Input array of shape (N,)
        params: Weights array of shape (10,)
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    
    # Define 10 centers uniformly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width for the Gaussian kernels
    sigma = 1.0
    
    # Compute the basis matrix
    # basis[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigma)^2)
    # We can vectorize this for speed
    # x shape: (N, 1)
    # centers shape: (1, 10)
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    # Calculate squared distances
    dist_sq = (x_expanded - c_expanded) ** 2
    
    # Calculate Gaussian values
    basis = np.exp(-0.5 * dist_sq / (sigma ** 2))
    
    # Linear combination: y = basis @ params
    return basis @ params

# EVOLVE-BLOCK-END
