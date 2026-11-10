<<<<<<< SEARCH
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev Polynomial Basis Expansion.
    
    Uses 10 Chebyshev polynomials of the first kind T_k(t) mapped to t = x/3.
    This provides an orthogonal (with respect to the weight 1/sqrt(1-t^2)) and 
    well-conditioned basis for approximating smooth functions on [-3, 3].
    
    The model is linear in parameters:
    y(x) = sum_{k=0}^{9} params[k] * T_k(x/3)
    
    BFGS will converge rapidly to the global optimum (least squares solution) 
    because the design matrix is well-conditioned compared to standard monomials 
    or mixed non-orthogonal bases.
    """
    # Map x from [-3, 3] to t in [-1, 1] for standard Chebyshev definition
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Linear combination
    # params shape (10,), T shape (10, N)
    # Result shape (N,)
    result = np.sum(params[:, np.newaxis] * T, axis=0)
    
    return result
=======
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Product of softplus-activated linear terms.
    
    y = p_0 * prod_{i=1}^{9} softplus(p_i * x + p_{i+1})
    
    This creates a piecewise-polynomial structure with learnable breakpoints
    to capture sharp transitions or kinks, while maintaining differentiability
    for BFGS optimization.
    """
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    
    # Scale x to [-1, 1] for better conditioning
    t = x / 3.0
    
    # p_0 is the overall scale
    scale = params[0]
    
    # For i = 1 to 9, we use p_i and p_{i+1}
    # softplus(z) = log(1 + exp(z))
    # Use stable computation: log(1 + exp(z)) = max(0, z) + log1p(exp(-|z|))
    
    result = np.ones_like(t, dtype=np.float64)
    
    for i in range(1, 10):
        z = params[i] * t + params[i + 1]
        # Stable softplus
        sp = np.maximum(0.0, z) + np.log1p(np.exp(-np.abs(z)))
        result = result * sp
    
    result = scale * result
    
    return result
>>>>>>> REPLACE
