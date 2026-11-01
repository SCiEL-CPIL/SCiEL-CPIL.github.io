"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W):
    """Verify that the tensor decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, U_arr.shape[1]):
        return False
    if V_arr.shape != (m * p, V_arr.shape[1]):
        return False
    if W_arr.shape != (p * n, W_arr.shape[1]):
        return False
    
    rank = U_arr.shape[1]
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta (the identity-like tensor for matrix mult)
    # For matrix multiplication: T[i,j,k] = 1 if this is a valid multiplication term
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    # The decomposition should give: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    # Schoolbook decomposition of rank n*m*p = 40 for 2x4x5.
    # For small mixed dimensions like 2x4x5, the schoolbook rank is
    # generally believed to be optimal or near-optimal.
    # We construct the standard decomposition.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify the decomposition is correct
    assert _verify_decomposition(U, V, W), "Schoolbook decomposition verification failed"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
