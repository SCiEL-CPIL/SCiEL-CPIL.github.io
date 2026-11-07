# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation with 10 parameters. We use a basis that can represent many common
    functional forms: polynomials, exponentials, logarithms, reciprocals, etc.
    
    The evaluator fits params via BFGS to minimize training MSE.
    We design a basis that is rich enough to approximate smooth functions on [0.1, 6].
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: exp(x)
    6: exp(-x)
    7: log(x)
    8: 1/x
    9: x * log(x)
    
    This covers polynomial trends, exponential growth/decay, logarithmic, and
    rational behaviors. The combination of these should approximate a wide range
    of smooth functions.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is within safe range for log and 1/x
    # x is drawn from [0.1, 6], so log(x) and 1/x are safe
    
    # Build basis
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    log_x = np.log(x)
    inv_x = 1.0 / x
    x_log_x = x * log_x
    
    # Combine with params
    result = (
        params[0] * 1.0 +
        params[1] * x +
        params[2] * x2 +
        params[3] * x3 +
        params[4] * x4 +
        params[5] * exp_x +
        params[6] * exp_neg_x +
        params[7] * log_x +
        params[8] * inv_x +
        params[9] * x_log_x
    )
    
    return result
# EVOLVE-BLOCK-END
