"""Rewritten program using a standardized Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a standardized Chebyshev polynomial basis.
    
    The model is linear in the 10 parameters: y = sum(params[i] * basis_i(x)).
    The basis functions are Chebyshev polynomials of the first kind, mapped 
    from the input domain [0.1, 6] to [-1, 1].
    
    Key design choices:
    - Chebyshev polynomials provide optimal approximation properties for 
      smooth functions on a bounded interval.
    - The basis is standardized (unit variance approx) to ensure the 
      Gram matrix is well-conditioned, allowing BFGS to converge reliably 
      to the global least-squares solution from the all-ones start.
    - This avoids the ill-conditioning issues of monomial bases (x^k) 
      and the potential collinearity of RBFs.
    """
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 100

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
        
    n = x.shape[0]
    if n == 0:
        return np.array([])

    # Map x from [0.1, 6.0] to t in [-1, 1]
    # t = (2x - (max + min)) / (max - min)
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Clamp t to [-1, 1] just in case of numerical drift outside domain
    # though the problem states inputs are in [0.1, 6]
    t = np.clip(t, -1.0, 1.0)
    
    # Initialize basis matrix B of shape (n, 10)
    # B[:, 0] = T_0(t) = 1
    # B[:, k] = T_k(t) for k >= 1
    B = np.zeros((n, 10), dtype=np.float64)
    
    # T_0(t) = 1
    B[:, 0] = 1.0
    
    if 10 > 1:
        # T_1(t) = t
        B[:, 1] = t
        
        if 10 > 2:
            # Recurrence: T_{k}(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
            # We store T_{k-1} and T_{k-2}
            T_k_minus_2 = np.ones(n, dtype=np.float64) # T_0
            T_k_minus_1 = t.copy()                     # T_1
            
            for k in range(2, 10):
                T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
                B[:, k] = T_k
                T_k_minus_2 = T_k_minus_1
                T_k_minus_1 = T_k
    
    # Standardize the basis columns to improve conditioning.
    # Chebyshev polynomials T_k(t) for k>=1 have mean 0 and variance 0.5 
    # with respect to the Chebyshev measure. 
    # Dividing by sqrt(2) makes them approximately unit variance.
    # T_0 is 1, so we leave it as is (or could scale, but 1 is standard).
    # This normalization helps BFGS by making the parameter scales more similar.
    if 10 > 1:
        B[:, 1:] /= np.sqrt(2.0)
        
    # Check for non-finite values in basis
    if not np.all(np.isfinite(B)):
        B = np.where(np.isfinite(B), B, 0.0)
        
    # Prediction: linear combination
    prediction = B @ p
    
    # Check for non-finite values in prediction
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
