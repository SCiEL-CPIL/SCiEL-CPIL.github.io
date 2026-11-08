import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

def _check_deadline(start_time, deadline=145.0):
    return (time.time() - start_time) < deadline

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the identity tensor for matrix multiplication
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def _trivial_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Return the known rank-30 trivial solution
    result = _trivial_decomposition()
    
    # Verify it's correct (optional, but good for debugging)
    # if not _verify_decomposition(result["U"], result["V"], result["W"], 2, 3, 5):
    #     raise ValueError("Trivial decomposition failed verification")
    
    return result
# EVOLVE-BLOCK-END
