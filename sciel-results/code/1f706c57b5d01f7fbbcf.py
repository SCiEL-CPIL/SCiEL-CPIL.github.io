# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rich basis of common functions.
    x in [-2, 2], params has length 10.
    
    Basis functions:
    0: 1 (constant)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: exp(x) - but scaled to avoid overflow issues; actually exp(x) for x in [-2,2] is fine
    7: tanh(x)
    8: |x|  -- non-differentiable at 0, but BFGS can handle it mostly; alternatively use x*sign or just x^2 for even
       Actually, let's use something smoother. Let's use x^4 / 4 or similar? 
       Let's stick with standard ones. |x| might cause issues with BFGS. 
       Let's use: 
       0: 1
       1: x
       2: x^2
       3: x^3
       4: sin(x)
       5: cos(x)
       6: exp(x)
       7: tanh(x)
       8: 1/(1+exp(-x))  (sigmoid) -- smooth
       9: x^4
    
    Wait, exp(x) for x in [-2,2] ranges from ~0.135 to ~7.39. That's fine.
    sigmoid is smooth.
    
    Let's ensure numerical stability.
    """
    # Precompute basis
    n = len(x)
    basis = np.empty((10, n), dtype=np.float64)
    
    basis[0] = 1.0
    basis[1] = x
    basis[2] = x * x
    basis[3] = x * x * x
    basis[4] = np.sin(x)
    basis[5] = np.cos(x)
    # exp(x) can be large but for x in [-2,2] it's fine
    basis[6] = np.exp(x)
    basis[7] = np.tanh(x)
    # sigmoid: 1/(1+exp(-x))
    # To avoid overflow in exp(-x) for x in [-2,2], it's safe. exp(2) ~ 7.39
    basis[8] = 1.0 / (1.0 + np.exp(-x))
    basis[9] = x * x * x * x
    
    # Linear combination: params (10,) dot basis (10, n)
    # result shape (n,)
    return basis @ params
# EVOLVE-BLOCK-END
