import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _tensor_error(params, n, m, p, rank):
    """Compute squared error of tensor reconstruction."""
    U = params[0:rank * n * m].reshape(rank, n * m)
    V = params[rank * n * m:2 * rank * n * m].reshape(rank, m * p)
    W = params[2 * rank * n * m:3 * rank * n * m].reshape(rank, p * n)
    
    # Target tensor: T[i, j, k] = 1 if output[i,k] depends on A[i,j]*B[j,k]
    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k in valid range
    # Wait, no. The tensor is defined such that the multiplication is:
    # C[i,k] = sum_t U[i*?+?, t] * V[?+?, t] * W[?+?, t]
    
    # The convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed over t gives the coefficient
    # for A[i,j]*B[j,k] in C[i,k]
    
    # The target is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i,j,k), the coefficient should be 1 if this term contributes to C[i,k]
    # which it always does (for valid i,j,k), so target[i,j,k] = 1
    
    # Reconstruct: for each (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # This should equal 1 for all i,j,k
    
    # But wait, that's not right either. Let me re-read the convention.
    # "W[k*n+i,r]" - so W is indexed by (k*n+i) where k is output column, i is output row
    # U[i*m+j, r] - i is input row of A, j is input column of A / input row of B
    # V[j*p+k, r] - j is input column of A / input row of B, k is input column of B / output column
    
    # The product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    # gives the coefficient multiplying A[i,j]*B[j,k] in the output C[i,k]
    
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
    # So the coefficient of A[i,j]*B[j,k] in C[i,k] is 1 for all valid i,j,k
    
    # Therefore, the target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let me verify with the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # for col corresponding to (i,j,k). Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = 1 when r corresponds to (i,j,k), else 0. So the sum is 1. Correct.
    
    # So we need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Compute this efficiently
    # For each r, the contribution is outer(U[:,r], V[:,r], W[:,r]) in some sense
    # Actually, for each r, we have vectors u_r (length n*m), v_r (length m*p), w_r (length p*n)
    # The tensor reconstructed is sum_r u_r ⊗ v_r ⊗ w_r (in the appropriate mode)
    
    # Let's compute the error as sum over all (i,j,k) of (sum_r ... - 1)^2
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
                error += (s - 1.0) ** 2
    return error

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with correct shapes: (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
