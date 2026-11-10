"""Symbolic regression with a flexible basis for hidden function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for precomputed basis normalization
_BASIS_NORMS = None

def _get_basis_norms():
    """Precompute RMS norms for the basis functions over [0.1, 6]."""
    global _BASIS_NORMS
    if _BASIS_NORMS is not None:
        return _BASIS_NORMS
    
    x_grid = np.linspace(0.1, 6.0, 2000)
    n_terms = 10
    
    # Define basis functions
    # 0: constant
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: exp(x)
    # 6: log(x)
    # 7: sin(x)
    # 8: cos(x)
    # 9: 1/x
    
    basis_grid = np.zeros((len(x_grid), n_terms))
    basis_grid[:, 0] = 1.0
    basis_grid[:, 1] = x_grid
    basis_grid[:, 2] = x_grid ** 2
    basis_grid[:, 3] = x_grid ** 3
    basis_grid[:, 4] = x_grid ** 4
    basis_grid[:, 5] = np.exp(x_grid)
    basis_grid[:, 6] = np.log(x_grid)
    basis_grid[:, 7] = np.sin(x_grid)
    basis_grid[:, 8] = np.cos(x_grid)
    basis_grid[:, 9] = 1.0 / x_grid
    
    # Compute RMS for each basis function
    rms = np.sqrt(np.mean(basis_grid ** 2, axis=0))
    rms = np.where(rms < 1e-12, 1.0, rms)
    
    _BASIS_NORMS = rms
    return _BASIS_NORMS


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible basis function approximation with 10 parameters.
    
    Uses a combination of polynomial, exponential, logarithmic, and trigonometric
    basis functions to approximate the hidden function. The basis is normalized
    by RMS to balance gradient magnitudes for BFGS convergence.
    """
    if len(x) == 0:
        return np.array([])
    
    n_terms = 10
    basis = np.zeros((len(x), n_terms))
    
    # 0: constant
    basis[:, 0] = 1.0
    # 1: x
    basis[:, 1] = x
    # 2: x^2
    basis[:, 2] = x ** 2
    # 3: x^3
    basis[:, 3] = x ** 3
    # 4: x^4
    basis[:, 4] = x ** 4
    # 5: exp(x)
    basis[:, 5] = np.exp(x)
    # 6: log(x)
    basis[:, 6] = np.log(x)
    # 7: sin(x)
    basis[:, 7] = np.sin(x)
    # 8: cos(x)
    basis[:, 8] = np.cos(x)
    # 9: 1/x
    basis[:, 9] = 1.0 / x
    
    # Normalize basis functions for better conditioning
    norms = _get_basis_norms()
    basis_normalized = basis / norms
    
    return basis_normalized @ params
# EVOLVE-BLOCK-END
