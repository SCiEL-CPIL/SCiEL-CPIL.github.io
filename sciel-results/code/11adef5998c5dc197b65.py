"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise linear basis (hat functions) with 10 fixed knots.
    This allows the model to capture local features and discontinuities
    that global polynomials might miss, while remaining linear in
    parameters for stable BFGS optimization.
    """
    # Ensure x is a 1D array
    x = np.asarray(x).flatten()
    
    # Domain of x is [0.1, 6.0]
    start = 0.1
    end = 6.0
    n_knots = 10
    
    # Define knots evenly spaced
    knots = np.linspace(start, end, n_knots)
    
    # Spacing between knots
    h = (end - start) / (n_knots - 1)
    
    # Compute basis functions
    # Shape of x: (N,)
    # Shape of knots: (10,)
    # We want basis shape: (N, 10)
    
    # x[:, np.newaxis] -> (N, 1)
    # knots[np.newaxis, :] -> (1, 10)
    # dist shape: (N, 10)
    dist = np.abs(x[:, np.newaxis] - knots[np.newaxis, :])
    
    # Hat function: max(0, 1 - dist/h)
    # This is 1 at the knot, 0 at distance h, and linear in between.
    basis = np.maximum(0.0, 1.0 - dist / h)
    
    # Linear combination of basis functions
    # basis: (N, 10)
    # params: (10,)
    # result: (N,)
    return basis @ params
# EVOLVE-BLOCK-END
