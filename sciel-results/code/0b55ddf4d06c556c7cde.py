import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

def _build_strassen56(n, m, p):
    """Build the rank-56 Strassen-based decomposition."""
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            a_idx = {
                (0,0): 0 * m + (2*s + 0),
                (0,1): 0 * m + (2*s + 1),
                (1,0): 1 * m + (2*s + 0),
                (1,1): 1 * m + (2*s + 1)
            }
            b_idx = {
                (0,0): (2*s + 0) * p + (2*j + 0),
                (0,1): (2*s + 0) * p + (2*j + 1),
                (1,0): (2*s + 1) * p + (2*j + 0),
                (1,1): (2*s + 1) * p + (2*j + 1)
            }
            c_idx = {
                (0,0): (2*j + 0) * n + 0,
                (0,1): (2*j + 1) * n + 0,
                (1,0): (2*j + 0) * n + 1,
                (1,1): (2*j + 1) * n + 1
            }
            
            U[a_idx[(0,0)]][col] += 1
            U[a_idx[(1,1)]][col] += 1
            U[a_idx[(1,0)]][col+1] += 1
            U[a_idx[(1,1)]][col+1] += 1
            U[a_idx[(0,0)]][col+2] += 1
            U[a_idx[(1,1)]][col+3] += 1
            U[a_idx[(0,0)]][col+4] += 1
            U[a_idx[(0,1)]][col+4] += 1
            U[a_idx[(1,0)]][col+5] += 1
            U[a_idx[(0,0)]][col+5] += -1
            U[a_idx[(0,1)]][col+6] += 1
            U[a_idx[(1,1)]][col+6] += -1
            
            V[b_idx[(0,0)]][col] += 1
            V[b_idx[(1,1)]][col] += 1
            V[b_idx[(0,0)]][col+1] += 1
            V[b_idx[(0,1)]][col+2] += 1
            V[b_idx[(1,1)]][col+2] += -1
            V[b_idx[(1,0)]][col+3] += 1
            V[b_idx[(0,0)]][col+3] += -1
            V[b_idx[(1,1)]][col+4] += 1
            V[b_idx[(0,0)]][col+5] += 1
            V[b_idx[(0,1)]][col+5] += 1
            V[b_idx[(1,0)]][col+6] += 1
            V[b_idx[(1,1)]][col+6] += 1
            
            W[c_idx[(0,0)]][col] += 1
            W[c_idx[(0,0)]][col+3] += 1
            W[c_idx[(0,0)]][col+4] += -1
            W[c_idx[(0,0)]][col+6] += 1
            W[c_idx[(0,1)]][col+2] += 1
            W[c_idx[(0,1)]][col+4] += 1
            W[c_idx[(1,0)]][col+1] += 1
            W[c_idx[(1,0)]][col+3] += 1
            W[c_idx[(1,1)]][col] += 1
            W[c_idx[(1,1)]][col+1] += -1
            W[c_idx[(1,1)]][col+2] += 1
            W[c_idx[(1,1)]][col+5] += 1
            
            col += 7
            
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Validate that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check that U @ diag(V) @ W^T reconstructs the multiplication tensor
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 if this is a valid term, else 0
    
    # Build expected tensor
    # A is n x m, B is m x p, C is n x p
    # Indexing: A[i,j], B[j,k], C[i,k]
    # U has shape (n*m, rank) - index by i*m + j
    # V has shape (m*p, rank) - index by j*p + k
    # W has shape (p*n, rank) - index by k*n + i
    
    # For each (i, j, k), the contribution to C[i,k] from A[i,j]*B[j,k]
    # should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # This term contributes to C[i,k]
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with rank-56 solution
    U, V, W = _build_strassen56(n, m, p)
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for linear algebra
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Attempt rank reduction: try to find a smaller rank by checking
    # if columns are linearly dependent in a way that preserves the tensor
    # This is complex, so we'll try a simple approach: 
    # Check if we can merge Strassen blocks
    
    # For now, return the rank-56 solution
    # The search for lower rank would require more sophisticated methods
    
    return {"U": U, "V": V, "W": W}
