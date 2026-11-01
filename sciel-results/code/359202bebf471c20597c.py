"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Check that the decomposition is exact
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if (i,j,k) is valid, 0 otherwise
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, that's not right. The tensor for matrix multiplication has T[i,j,k] = 1
    # when multiplying A[i,j] * B[j,k] to get C[i,k].
    # Actually, the standard matrix multiplication tensor has:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Let's verify: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _try_pattern_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-minimal decomposition using pattern-based approach."""
    start_time = time.time()
    
    # Start with the schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map r to (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i * m * p + j * p + k < rank:
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    # Try to reduce rank by merging terms
    # For each pair of terms, check if they can be merged
    current_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try removing terms one by one
    for r_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Try removing term r_remove
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Zero out the column for r_remove
        for i in range(len(U_test)):
            U_test[i][r_remove] = 0
        for i in range(len(V_test)):
            V_test[i][r_remove] = 0
        for i in range(len(W_test)):
            W_test[i][r_remove] = 0
        
        # Check if still valid
        if _validate_decomposition(U_test, V_test, W_test, n, m, p):
            # Remove the column
            U_test = [row[:r_remove] + row[r_remove+1:] for row in U_test]
            V_test = [row[:r_remove] + row[r_remove+1:] for row in V_test]
            W_test = [row[:r_remove] + row[r_remove+1:] for row in W_test]
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with rank n*m*p
    rank = n * m * p
    
    # Try to find a better decomposition
    result = _try_pattern_decomposition(n, m, p, rank, deadline)
    
    # Validate the result
    if _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
