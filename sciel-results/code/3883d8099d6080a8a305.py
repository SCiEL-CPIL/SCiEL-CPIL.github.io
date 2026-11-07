"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Construct cubic polynomial basis in 2D
    # Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    # params length is 10
    
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Create basis matrix
    # Shape: (N, 10)
    # Column 0: 1
    # Column 1: x
    # Column 2: y
    # Column 3: x^2
    # Column 4: x*y
    # Column 5: y^2
    # Column 6: x^3
    # Column 7: x^2*y
    # Column 8: x*y^2
    # Column 9: y^3
    
    # Vectorized construction
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = y
    term_3 = x**2
    term_4 = x * y
    term_5 = y**2
    term_6 = x**3
    term_7 = x**2 * y
    term_8 = x * y**2
    term_9 = y**3
    
    # Stack into a matrix
    # np.column_stack is efficient
    basis = np.column_stack((term_0, term_1, term_2, term_3, term_4, term_5, term_6, term_7, term_8, term_9))
    
    # Compute prediction
    # params is a 1D array of length 10
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
