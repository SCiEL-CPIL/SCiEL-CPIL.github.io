"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for consistent shape handling
    x_flat = np.asarray(x).flatten()
    
    # Scale x from [-2, 2] to [-1, 1] for stable Chebyshev polynomials
    # This keeps the basis functions bounded by 1, aiding optimization stability
    t = 0.5 * x_flat
    
    # Number of parameters available
    n_params = len(params)
    
    # Compute Chebyshev polynomials T_k(t) for k=0 to n_params-1
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Initialize basis matrix
    # We will store T_k in columns
    basis = np.zeros((len(x_flat), n_params))
    
    if n_params > 0:
        basis[:, 0] = 1.0
        if n_params > 1:
            basis[:, 1] = t
            
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Compute prediction: sum of params[k] * T_k(t)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
