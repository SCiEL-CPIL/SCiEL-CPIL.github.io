import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev series expansion.
    The input x is scaled to [-1, 1] to ensure numerical stability and
    well-conditioning for the BFGS optimizer.
    
    We use 10 parameters corresponding to Chebyshev polynomials T_0 through T_9.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1] for Chebyshev polynomials
    # Original domain is [-3, 3]
    u = x / 3.0
    
    # Number of basis functions (parameters)
    n_basis = 10
    n_points = x.size
    
    # Initialize the basis matrix
    # Shape: (n_points, n_basis)
    T = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # T_0(u) = 1
    T[:, 0] = 1.0
    
    # T_1(u) = u
    if n_basis > 1:
        T[:, 1] = u
        
    # Recurrence: T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
    if n_basis > 2:
        for k in range(2, n_basis):
            T[:, k] = 2.0 * u * T[:, k-1] - T[:, k-2]
            
    # Compute the linear combination
    # params shape: (10,)
    # T shape: (n_points, 10)
    # Result shape: (n_points,)
    return T @ params
