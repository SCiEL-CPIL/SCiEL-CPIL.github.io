"""
Strategy: Local Gaussian Radial Basis Function (RBF) Basis.

Instead of using global polynomial or Chebyshev basis functions, this model
uses a linear combination of 10 Gaussian "bumps" centered at fixed locations
across the input domain [-2, 2].

Model:
y(x) = sum_{i=0}^{9} c_i * exp(-(x - c_i)^2 / (2 * sigma^2))

Where:
- c_i are fixed centers: -2.0, -1.6, ..., 2.0
- sigma is a fixed width: 0.4
- c_i are the 10 parameters optimized by BFGS.

Why this works better:
1. Locality: Each term only affects the function near its center, allowing
   independent fitting of local features (peaks, dips, sharp changes) without
   global oscillations (Runge's phenomenon) seen in high-degree polynomials.
2. Smoothness: Gaussian functions are infinitely differentiable, ensuring
   stable gradients for BFGS.
3. Flexibility: A sum of Gaussians can approximate a wide class of smooth
   and piecewise-smooth functions, including those with non-polynomial
   behavior like |x| or sigmoid-like transitions, more effectively than
   low-degree polynomials.
"""
import numpy as np

def _build_rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Build the 10-column Gaussian RBF basis matrix.
    
    Args:
        x: Input array of shape (N,).
        
    Returns:
        phi: Basis matrix of shape (N, 10).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    
    # Define 10 centers uniformly spaced in [-2, 2]
    # Using 10 points: -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0
    # Wait, 10 points from -2 to 2 inclusive is step 0.444...
    # Let's use np.linspace(-2, 2, 10)
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for the Gaussians.
    # Domain width is 4. 10 centers. Average spacing is ~0.44.
    # Sigma = 0.4 ensures good overlap for smooth interpolation but maintains locality.
    sigma = 0.4
    
    # Compute basis values
    # phi[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigma^2))
    # Reshape x to (N, 1) and centers to (1, 10) for broadcasting
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    phi = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    return phi

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute the predicted y values using the Gaussian RBF basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 coefficients.
        
    Returns:
        predictions: Array of shape (N,).
    """
    x = np.asarray(x, dtype=float).ravel()
    params = np.asarray(params, dtype=float).ravel()
    
    # Ensure params is length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Build basis and compute linear combination
    phi = _build_rbf_basis(x)
    result = phi @ params
    
    # Safety check for numerical stability
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
