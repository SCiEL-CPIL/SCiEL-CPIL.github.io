import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis.
    
    Chebyshev polynomials are orthogonal on the interval [-1, 1] and provide
    a well-conditioned basis for polynomial approximation. By mapping the input
    domain [-3, 3] to [-1, 1], we can use a degree-9 Chebyshev series (10 terms)
    to fit the function. This global, smooth basis is structurally different from
    local RBFs or splines and is typically very well-conditioned for BFGS,
    allowing it to converge reliably to the best polynomial approximation of the
    underlying function.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    p = params[:10]
    
    # Ensure finite output
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((n, 10))
    T[:, 0] = 1.0
    if n > 0:
        T[:, 1] = t
        for k in range(2, 10):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
            
    # Prediction
    y_pred = T @ p
    
    return y_pred
