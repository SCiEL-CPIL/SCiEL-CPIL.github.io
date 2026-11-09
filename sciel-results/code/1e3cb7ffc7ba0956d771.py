"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a 9th-degree polynomial basis.
    The basis functions are 1, x, x^2, ..., x^9.
    This allows the model to fit smooth non-linear functions well.
    """
    # Ensure x is a 1D array for vander
    x = np.asarray(x, dtype=np.float64)
    
    # Create the Vandermonde matrix: columns are [1, x, x^2, ..., x^9]
    # N=10 corresponds to degree 9
    # increasing=True means the first column is x^0, last is x^9
    basis = np.vander(x, N=10, increasing=True)
    
    # Compute the prediction: sum of params[i] * basis[:, i]
    # basis shape: (N_points, 10)
    # params shape: (10,)
    predictions = basis @ params
    
    # Safety check: if any predictions are non-finite, return a large constant
    # to avoid invalidating the fit, though this shouldn't happen for x in [0.1, 6]
    if not np.all(np.isfinite(predictions)):
        return np.full_like(predictions, 1e6)
        
    return predictions
# EVOLVE-BLOCK-END
