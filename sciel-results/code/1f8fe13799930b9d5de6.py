"""Symbolic regression with 2 inputs using a rich polynomial basis + BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin before 120s limit

    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Construct a rich mixed basis with 10 terms.
    # The domain is [-1, 1], so we use polynomial and trigonometric terms.
    # We include:
    # - Constant
    # - Linear terms: x, y
    # - Quadratic terms: x^2, y^2, xy
    # - Cubic terms: x^3, y^3
    # - Interaction: x^2*y
    # - Trigonometric: sin(x), sin(y), cos(x+y)
    # 
    # This covers both polynomial and oscillatory behaviors.
    
    # Term 0: Constant (1)
    term_0 = np.ones_like(x)
    
    # Term 1: x
    term_1 = x
    
    # Term 2: y
    term_2 = y
    
    # Term 3: x^2
    term_3 = x ** 2
    
    # Term 4: y^2
    term_4 = y ** 2
    
    # Term 5: xy
    term_5 = x * y
    
    # Term 6: x^3
    term_6 = x ** 3
    
    # Term 7: y^3
    term_7 = y ** 3
    
    # Term 8: x^2*y
    term_8 = x ** 2 * y
    
    # Term 9: sin(x+y) - interaction trigonometric
    term_9 = np.sin(x + y)
    
    # Stack terms into a matrix of shape (n_points, 10)
    basis = np.column_stack([
        term_0, term_1, term_2, term_3, term_4,
        term_5, term_6, term_7, term_8, term_9
    ])
    
    # The model is linear in params: pred = basis @ params
    # BFGS will optimize params to minimize MSE.
    # Since it's linear, this is effectively a least squares problem,
    # but BFGS is robust enough to find it quickly from any start.
    
    predictions = basis @ params
    
    # Ensure output is finite and correct shape
    if not np.all(np.isfinite(predictions)):
         # Fallback to zero if something went wrong (should not happen)
         return np.zeros_like(x)
        
    return predictions


# EVOLVE-BLOCK-END
