"""Symbolic regression with 1 input(s) using a hybrid Chebyshev-RBF-Fourier basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - 4 Chebyshev polynomials (mapped to [-1, 1]) for global smooth structure.
    - 3 Gaussian RBFs (centers at 1.5, 3.0, 4.5) for local flexibility.
    - 2 Fourier terms (sin, cos) at one frequency over the domain for oscillatory components.
    - 1 Exponential decay term exp(-x) for decaying behavior.
    
    This basis is chosen to cover a wide range of common functional forms:
    Chebyshev terms capture global trends and oscillatory behavior efficiently,
    RBFs capture local bumps, peaks, and sharp features, Fourier terms capture
    periodic components, and the exponential term captures decay/growth.
    
    The input x is mapped to t in [-1, 1] for the Chebyshev basis.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # Part 1: Chebyshev polynomials (4 terms: T_0 to T_3)
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (min+max)) / (max-min)
    # min=0.1, max=6.0
    t = (2.0 * x - 6.1) / 5.9
    t = np.clip(t, -1.0, 1.0)
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 4):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Part 2: Gaussian RBFs (3 terms)
    # Centers evenly spaced: 1.5, 3.0, 4.5
    # Bandwidth sigma = 0.8
    centers = np.array([1.5, 3.0, 4.5])
    sigma = 0.8
    
    # Compute RBFs: exp(-(x - c)^2 / (2 * sigma^2))
    # x is (N,), centers is (3,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    basis[:, 4:7] = rbf_terms
    
    # Part 3: Fourier terms (2 terms)
    # One period over the domain [0.1, 6]
    # Frequency: 2*pi / (6.0 - 0.1) ≈ 2*pi / 5.9
    freq = 2.0 * np.pi / 5.9
    basis[:, 7] = np.sin(freq * x)
    basis[:, 8] = np.cos(freq * x)
    
    # Part 4: Exponential decay (1 term)
    basis[:, 9] = np.exp(-x)
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
# EVOLVE-BLOCK-END
