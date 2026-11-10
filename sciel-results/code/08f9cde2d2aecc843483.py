"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 Chebyshev polynomials.
    
    The input x is in [-2, 2]. We map it to t in [-1, 1] for Chebyshev stability.
    The model is: y = sum_{k=0}^{9} p_k * T_k(t)
    
    This is a linear model in parameters, ensuring the optimization landscape is convex.
    BFGS will efficiently find the global least-squares solution within this subspace.
    Chebyshev polynomials provide a well-conditioned basis that can approximate a wide
    variety of smooth functions better than a simple linear model.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Ensure p is a flat array of length 10
    p = np.asarray(p, dtype=np.float64).flatten()
    if p.shape[0] != 10:
        # Safety check, though evaluator guarantees length 10
        p = np.resize(p, 10)
        
    # Map x from [-2, 2] to [-1, 1]
    t = x * 0.5
    
    # Compute Chebyshev polynomials T_0 to T_9
    # Using recurrence: T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)
    
    # Initialize T_0 and T_1
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # If we only need up to T_9, we can compute them iteratively
    # We accumulate the sum to avoid storing all T_k if memory was a concern, 
    # but N is small (200), so storing is fine.
    
    # Let's just compute the sum directly to be efficient and clean
    y = p[0] * T0 + p[1] * T1
    
    if len(p) > 2:
        T_prev = T0
        T_curr = T1
        for k in range(2, 10):
            T_next = 2.0 * t * T_curr - T_prev
            y += p[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return y
# EVOLVE-BLOCK-END
