import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression using a Hybrid Basis (Polynomial + Exponential + Log + Rational + Sinusoidal).
    
    Strategy:
    - Use a hybrid basis that is linear in parameters.
    - This ensures the optimization landscape is convex (quadratic loss), 
      allowing BFGS to converge reliably to the global minimum.
    - The basis includes:
      1. Polynomials (1, x, x^2, x^3, x^4)
      2. Exponentials (exp(-x), exp(x))
      3. Logarithm (log(x))
      4. Reciprocal (1/x)
      5. Sinusoidal (sin(x))
    - This combination is flexible enough to approximate a wide variety 
      of smooth functions on the domain [0.1, 6].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    n = len(x)
    if n == 0:
        return np.array([])
        
    # Compute basis functions
    # 1. Constant
    b0 = np.ones(n)
    
    # 2. Linear
    b1 = x.copy()
    
    # 3. Quadratic
    b2 = x ** 2
    
    # 4. Cubic
    b3 = x ** 3
    
    # 5. Quartic
    b4 = x ** 4
    
    # 6. Exponential decay
    b5 = np.exp(-x)
    
    # 7. Exponential growth
    b6 = np.exp(x)
    
    # 8. Logarithm
    # x is in [0.1, 6], so log is safe
    b7 = np.log(x)
    
    # 9. Reciprocal
    b8 = 1.0 / x
    
    # 10. Sinusoidal
    b9 = np.sin(x)
    
    # Stack basis functions into a matrix (N, 10)
    # result = B @ p
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute linear combination
    result = B @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # If any non-finite values, return zeros or handle gracefully
        # However, with x in [0.1, 6], exp(x) is max ~403, 1/x is max 10, log(x) is max ~1.8.
        # Non-finite values should not occur unless params are huge, but BFGS keeps them bounded.
        # If they do, returning zeros is a safe fallback.
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
