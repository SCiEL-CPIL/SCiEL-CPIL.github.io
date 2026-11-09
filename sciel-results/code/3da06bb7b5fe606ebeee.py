"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise-linear spline basis.
    
    The domain [-3, 3] is divided into 9 equal intervals, creating 10 knots.
    We use 10 'hat' functions (triangular basis functions) centered at these knots.
    This local basis ensures that each parameter primarily influences the function
    value near its corresponding knot, leading to a well-conditioned design matrix
    for the BFGS optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Domain bounds
    x_min = -3.0
    x_max = 3.0
    
    # Number of basis functions (matches params length)
    n_basis = 10
    
    # Define knots: 10 points evenly spaced in [-3, 3]
    # Knots: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 3
    knots = np.linspace(x_min, x_max, n_basis)
    
    # Interval width
    h = (x_max - x_min) / (n_basis - 1)
    
    # Initialize output
    y = np.zeros_like(x, dtype=np.float64)
    
    # Construct hat functions
    for i in range(n_basis):
        # Center of the hat function
        c = knots[i]
        
        # Left and right boundaries of the support
        # For interior points, support is [c-h, c+h]
        # For boundary points, support is [c, c+h] or [c-h, c]
        
        # Calculate the hat function value for all x
        # T_i(x) = max(0, 1 - |x - c| / h)
        
        # We can compute this vectorized
        # dist = np.abs(x - c)
        # hat = np.maximum(0.0, 1.0 - dist / h)
        
        # However, standard hat functions on a uniform grid with knots at the centers
        # of intervals or at the nodes?
        # If knots are nodes, the hat function is 1 at knot i, 0 at all other knots.
        # The formula 1 - |x-c|/h works if the neighbors are at distance h.
        
        # Let's verify:
        # Knot 0: -3. Hat is 0 at -3-h (out of bounds), 1 at -3, 0 at -3+h (-2.4).
        # Formula: 1 - |-3 - (-3)|/h = 1. Correct.
        # At -2.4: 1 - |-2.4 - (-3)|/h = 1 - 0.6/0.6 = 0. Correct.
        
        dist = np.abs(x - c)
        hat = np.maximum(0.0, 1.0 - dist / h)
        
        y += params[i] * hat
        
    return y
# EVOLVE-BLOCK-END
