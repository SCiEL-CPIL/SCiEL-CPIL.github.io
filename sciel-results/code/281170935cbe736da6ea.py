import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Degree 9 Chebyshev Series.
    
    The target function is defined on x in [-2, 2]. We map this to t in [-1, 1]
    where Chebyshev polynomials are orthogonal and well-conditioned.
    
    y(x) = sum_{k=0}^{9} c_k * T_k(t), where t = x / 2.
    
    Parameters:
    - params[0..9]: Coefficients c_0 to c_9.
    
    This approach is materially different from the monomial polynomial (current)
    because Chebyshev polynomials minimize the maximum error for a given degree
    (minimax property) and provide a much better-conditioned basis for BFGS
    optimization, especially for smooth functions. It differs from Gaussian
    mixtures by being a global basis, ensuring no local minima related to
    center placement.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Extract coefficients
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Evaluate Chebyshev polynomials T_k(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    T_k_minus_2 = np.ones_like(t)
    T_k_minus_1 = t.copy()
    
    y = c0 * T_k_minus_2 + c1 * T_k_minus_1
    
    if len(params) > 2:
        for k in range(2, 10):
            T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
            y += params[k] * T_k
            T_k_minus_2 = T_k_minus_1
            T_k_minus_1 = T_k
            
    return y
