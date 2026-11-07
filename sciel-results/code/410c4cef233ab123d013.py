import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a Legendre polynomial series of degree 9.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * P_i(x/3.0)
    
    Legendre polynomials P_i are orthogonal on [-1, 1] with respect to the uniform weight.
    Since the data is drawn uniformly from [-3, 3] and the loss is MSE (L2),
    Legendre polynomials provide a better-conditioned and theoretically optimal basis
    for least-squares fitting compared to Chebyshev polynomials (which are optimal for minimax).
    """
    # Scale input to [-1, 1]
    z = x / 3.0
    
    N = 10
    # Initialize basis matrix
    # We can compute row by row or column by column.
    # Column by column is easier with recurrence.
    
    P_prev = np.ones_like(z)  # P_0
    P_curr = z.copy()         # P_1
    
    # Store in a list or matrix
    # Since N is small (10), we can just build the vector for each polynomial
    # But we need to return a linear combination.
    
    # Let's build the basis matrix for clarity and speed (vectorized ops)
    # Actually, we can just accumulate the result.
    
    y = params[0] * P_prev + params[1] * P_curr
    
    for n in range(1, N - 1):
        # Compute P_{n+1}
        # P_{n+1} = ((2n+1)*z*P_n - n*P_{n-1}) / (n+1)
        P_next = ((2 * n + 1) * z * P_curr - n * P_prev) / (n + 1)
        
        y += params[n + 1] * P_next
        
        P_prev = P_curr
        P_curr = P_next
        
    return y
