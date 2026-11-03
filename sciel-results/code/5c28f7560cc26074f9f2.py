"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist(), 'rank': rank_sb}
    
    # Try to find a lower rank decomposition
    # For each candidate rank, try to construct a valid decomposition
    # We'll use a simple approach: try ranks from a lower bound up
    
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But realistically, for matrix multiplication, the rank is at least n*m*p / min(n,m,p) or similar
    # Let's try a range of ranks
    
    # Define the tensor T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] has this structure
    # when we consider the tensor itself, not specific matrices)
    # Actually, the tensor is T[i,j,k] which represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 if the product contributes to C[i,k], which is always the case.
    # Wait, no. The matrix multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 3-way tensor decomposition problem.
    
    # For now, let's just try to verify the schoolbook and return it.
    # A proper search would require solving a hard optimization problem.
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
