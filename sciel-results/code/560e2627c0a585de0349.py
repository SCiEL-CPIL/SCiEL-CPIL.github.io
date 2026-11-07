"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear combination of Orthonormal 
    Legendre polynomials. 
    Legendre polynomials are orthogonal with respect to the uniform distribution 
    on [-1, 1], which matches the input distribution. Scaling them to be 
    orthonormal improves the condition number of the design matrix, aiding 
    BFGS optimization.
    """
    # Ensure x is a 1D numpy array for consistent processing
    x = np.asarray(x, dtype=float).flatten()
    
    # Safety check: if inputs are non-finite, return zeros to avoid NaN propagation
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    # Normalize input to [-1, 1]
    # x is in [-3, 3], so t = x/3 is in [-1, 1]
    t = x / 3.0
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Create a matrix to hold the 10 orthonormal basis functions
    # Basis functions: sqrt(2k+1) * P_k(t)
    basis = np.zeros((n_points, 10))
    
    # P_0(t) = 1
    # Orthonormal: sqrt(1) * P_0(t) = 1
    basis[:, 0] = 1.0
    
    # P_1(t) = t
    # Orthonormal: sqrt(3) * P_1(t) = sqrt(3) * t
    basis[:, 1] = np.sqrt(3.0) * t
    
    # Compute higher order Legendre polynomials using recurrence relation:
    # (n+1) P_{n+1}(t) = (2n+1) t P_n(t) - n P_{n-1}(t)
    # We store P_k(t) in a temporary array to compute the next one,
    # then scale and store in basis.
    
    # We need P_0 and P_1 to start.
    # P_0 = 1
    # P_1 = t
    
    P_prev = 1.0 * np.ones_like(t) # P_0
    P_curr = t                      # P_1
    
    for k in range(2, 10):
        # Calculate P_k
        # P_k = ((2k-1) * t * P_{k-1} - (k-1) * P_{k-2}) / k
        P_next = ((2.0 * k - 1.0) * t * P_curr - (k - 1.0) * P_prev) / k
        
        # Scale to orthonormal: sqrt(2k+1) * P_k
        basis[:, k] = np.sqrt(2.0 * k + 1.0) * P_next
        
        # Update for next iteration
        P_prev = P_curr
        P_curr = P_next
            
    # Return the linear combination of basis functions weighted by params
    # params is expected to be a 1D array of length 10
    return basis @ params
# EVOLVE-BLOCK-END
