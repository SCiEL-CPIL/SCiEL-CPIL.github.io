import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a scaled monomial basis.
    
    The basis functions are 1, u, u^2, ..., u^9 where u = x/2.
    This scaling maps the input domain [-2, 2] to [-1, 1], which improves
    the numerical conditioning of the optimization problem for BFGS.
    """
    # Scale input to [-1, 1] to improve conditioning
    u = 0.5 * x
    
    # Compute powers of u up to degree 9
    # We construct the design matrix B of shape (N, 10)
    N = len(x)
    B = np.ones((N, 10))
    
    # B[:, 0] is already 1.0
    # Fill in the rest
    # Using np.power or iterative multiplication
    # Iterative is faster and avoids repeated power calculations
    current_power = np.ones(N)
    for k in range(1, 10):
        current_power = current_power * u
        B[:, k] = current_power
        
    # Linear combination
    return B @ params
