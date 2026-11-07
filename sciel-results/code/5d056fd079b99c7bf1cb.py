"""
Symbolic regression using a diverse basis of polynomial, trigonometric, and
activation-like functions. This hybrid basis provides flexibility to capture
various function shapes while remaining linear in parameters for reliable BFGS
convergence.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points in [-2, 2]
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Build a diverse basis of 10 functions
    # 1. Constant
    # 2. Linear
    # 3. Quadratic
    # 4. Cubic
    # 5. x^4
    # 6. sin(x)
    # 7. cos(x)
    # 8. tanh(x)
    # 9. exp(-x^2/2) (Gaussian bump)
    # 10. |x| (absolute value)
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.tanh(x),
        np.exp(-x**2 / 2.0),
        np.abs(x)
    ])
    
    pred = basis.dot(params)
    
    return pred
# EVOLVE-BLOCK-END
