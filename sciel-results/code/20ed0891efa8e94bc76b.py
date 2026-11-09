import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a mixed basis that includes both Chebyshev polynomials (for smooth polynomial-like behavior)
    and common non-polynomial functions (exp, log, trig, rational) to capture a wider range of target functions.
    
    Basis (10 features):
      0: 1
      1: x
      2: x^2
      3: x^3
      4: exp(x)
      5: exp(-x)
      6: log(1 + x)
      7: sin(x)
      8: cos(x)
      9: 1/(1 + x)
    """
    x = np.asarray(x, dtype=float)
    
    # Compute basis functions
    basis = np.ones_like(x)  # 0: 1
    
    # 1: x
    x1 = x
    basis = np.column_stack((basis, x1))
    
    # 2: x^2
    x2 = x1 * x1
    basis = np.column_stack((basis, x2))
    
    # 3: x^3
    x3 = x2 * x1
    basis = np.column_stack((basis, x3))
    
    # 4: exp(x)
    exp_x = np.exp(x)
    basis = np.column_stack((basis, exp_x))
    
    # 5: exp(-x)
    exp_neg_x = np.exp(-x)
    basis = np.column_stack((basis, exp_neg_x))
    
    # 6: log(1 + x)
    log_1px = np.log1p(x)
    basis = np.column_stack((basis, log_1px))
    
    # 7: sin(x)
    sin_x = np.sin(x)
    basis = np.column_stack((basis, sin_x))
    
    # 8: cos(x)
    cos_x = np.cos(x)
    basis = np.column_stack((basis, cos_x))
    
    # 9: 1/(1 + x)
    inv_1px = 1.0 / (1.0 + x)
    basis = np.column_stack((basis, inv_1px))
    
    # Linear combination
    y = basis @ params
    
    return y
