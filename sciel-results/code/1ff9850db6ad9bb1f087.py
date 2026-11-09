"""
Strategy: Chebyshev Polynomial Basis.

This strategy replaces the local Radial Basis Functions (RBFs) with a global basis of 
Chebyshev polynomials of the first kind. 

Rationale:
1. Better Conditioning: Chebyshev polynomials are orthogonal on the interval [-1, 1] 
   (with a specific weight). When mapped from [-3, 3] to [-1, 1], they provide a 
   well-conditioned basis for polynomial approximation. This is significantly better 
   than monomials (which are ill-conditioned) and often better than RBFs for smooth 
   functions, as RBFs can lead to redundant features if centers are too close.
2. Minimax Property: Chebyshev expansions minimize the maximum error for polynomial 
   approximations, which helps in capturing the global shape of the function more 
   uniformly than RBFs might, especially if the function is smooth but non-local.
3. Linear in Parameters: The model remains linear in the weights, ensuring that the 
   evaluator's BFGS optimizer can find the global least-squares solution efficiently.
4. Universal Approximation: Polynomials are universal approximators on compact intervals. 
   With 10 degrees of freedom (up to degree 9), it can fit a wide range of smooth 
   functions.

The basis functions are $T_k(2x/3 - 1)$ for $k=0, \dots, 9$.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the Chebyshev polynomial expansion.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    
    # Map x from [-3, 3] to t in [-1, 1]
    # t = (x - 0) / 3 = x/3
    # Wait, standard mapping: t = (2x - (max+min)) / (max-min)
    # max=3, min=-3.
    # t = (2x - 0) / 6 = x/3.
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    N = len(x)
    basis = np.zeros((N, 10))
    
    # T_0
    basis[:, 0] = 1.0
    
    # T_1
    basis[:, 1] = t
    
    # Recurrence for k=2 to 9
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Linear combination: y = basis @ params
    return basis @ params

# EVOLVE-BLOCK-END
