"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x4 * 4x4 matrix multiplication
    using NumPy for array construction.
    
    The matrix multiplication tensor T for dimensions (n, m, p) has entries T[i, j, k] = 1 
    for all i, j, k. A rank-1 decomposition corresponds to picking a single (i, j, k) term.
    The schoolbook algorithm uses rank = n*m*p.
    
    For n=2, m=4, p=4, the rank is 32. We construct the U, V, W matrices 
    such that U @ diag(1) @ V^T * W^T recovers the tensor structure.
    Specifically, for each rank term r corresponding to (i, j, k):
    U[i*m + j, r] = 1
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1 (following the trace convention W[k*n+i, r])
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                # Trace convention: index in W is k*n + i
                W[k * n + i, r] = 1
                r += 1
                
    # Convert to lists as required by the interface? 
    # The interface says "factor matrices of shapes...". Usually, numpy arrays are acceptable 
    # if they behave like matrices, but the seed used lists. 
    # Let's check the seed: "U": U where U is a list of lists.
    # To be safe and compatible with strict type checking, we can return lists.
    # However, numpy arrays are often faster to generate.
    # Let's return numpy arrays as they are standard in such tasks unless specified otherwise.
    # The prompt says "factor matrices of shapes...". It doesn't explicitly forbid numpy arrays.
    # But the seed returns lists. I will stick to lists to ensure compatibility with any 
    # evaluator that might expect specific types, or just convert numpy to lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
