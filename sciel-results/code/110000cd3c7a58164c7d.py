import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use Legendre polynomials as the basis.
    Map x from [0.1, 6] to u in [-1, 1].
    Compute P_0 to P_9.
    Return linear combination with params.
    Legendre polynomials are orthogonal with respect to the uniform distribution,
    which matches the data generation process.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x to [-1, 1]
    # x_min = 0.1, x_max = 6.0
    # u = (2x - (x_max + x_min)) / (x_max - x_min)
    u = (2.0 * x - 6.1) / 5.9
    
    # Initialize output
    y = np.zeros_like(x)
    
    # P_0 = 1
    P_prev = np.ones_like(x)
    y += params[0] * P_prev
    
    if len(params) > 1:
        # P_1 = u
        P_curr = u
        y += params[1] * P_curr
        
        for k in range(2, 10):
            # Recurrence: (k) P_k = (2k-1) u P_{k-1} - (k-1) P_{k-2}
            # P_k = ((2k-1) u P_{k-1} - (k-1) P_{k-2}) / k
            P_next = ((2.0 * k - 1.0) * u * P_curr - (k - 1.0) * P_prev) / k
            y += params[k] * P_next
            P_prev = P_curr
            P_curr = P_next
            
    return y
