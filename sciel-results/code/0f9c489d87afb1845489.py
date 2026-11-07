import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a degree-9 Chebyshev polynomial basis.
    
    The model is linear in the parameters, ensuring that the optimization
    landscape (MSE) is convex. This allows BFGS to find the global minimum
    reliably, avoiding local minima issues common in non-linear models.
    
    The input x is scaled to [-1, 1] to utilize the stable properties of
    Chebyshev polynomials on that domain.
    
    Parameters (10 total):
    p0 ... p9: Coefficients for Chebyshev polynomials T0(t) ... T9(t)
    where t = x / 2.
    """
    # Scale x to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T0 to T9
    # T0(t) = 1
    # T1(t) = t
    # Tn(t) = 2*t*Tn-1(t) - Tn-2(t)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Linear combination: y = sum(params[i] * T[i, :])
    # Equivalent to T.dot(params)
    y = np.dot(T, params)
    
    return y
