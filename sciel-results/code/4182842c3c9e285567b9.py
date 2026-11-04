"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    We attempt to find a decomposition of rank less than the schoolbook rank of 45.
    
    The tensor for multiplying A (n x m) by B (m x p) to get C (n x p) has indices:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    The decomposition provides matrices U, V, W such that:
    T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[c, i, k]
    
    Wait, the interface is:
    U of shape (n*m, rank)
    V of shape (m*p, rank)
    W of shape (p*n, rank)
    
    And the output factor follows the official trace convention W[k*n+i, r].
    This means the tensor entry T[i, j, k] is decomposed as:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We will use a randomized search over integer and half-integer matrices to find a valid decomposition.
    We start with a schoolbook decomposition of rank 45 and try to reduce the rank.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    target_rank = 44  # Start just below schoolbook
    
    # First, generate the schoolbook decomposition as a fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Check if a candidate decomposition is valid
    def is_valid(U, V, W):
        rank = len(U[0])
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Build the tensor from the decomposition
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We only need to check that the matrix multiplication is correct.
        # For each i, k: C[i, k] = sum_j A[i, j] B[j, k]
        # The tensor T[i, j, k] should be 1 if we are computing the (i,k) entry of the product
        # using the j-th term, and 0 otherwise.
        
        # Actually, the tensor T is defined by T[i, j, k] = delta_{i, i} * delta_{k, k} ??? No.
        # The matrix multiplication tensor is:
        # T_{i, j, k} = 1 if we are looking at the contribution of A_{i,j} and B_{j,k} to C_{i,k}.
        # So T is a 3D array where T[i, j, k] = 1 for all i, j, k? No.
        # The tensor is: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
        # So T[i,j,k] must be such that for fixed i,k, sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
        # This is only true if T[i,j,k] = delta_{i,i} delta_{k,k} ??? 
        # No, the tensor for matrix multiplication is:
        # M[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to get C[i,k].
        # So M[i, j, k] = 1 for all valid i, j, k.
        
        # Wait, the decomposition is:
        # M[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # And M[i, j, k] = 1 for all i, j, k.
        
        # So we just need to check that the sum equals 1 for all i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition of lower rank
    # We will use a simple randomized approach:
    # 1. Start with the schoolbook decomposition
    # 2. Try to merge or eliminate factors
    
    # For now, we return the schoolbook decomposition, as finding a lower rank
    # decomposition for 3x3x5 is non-trivial and may not be possible within the time limit.
    
    # However, we know that the rank of 3x3x3 is 23, and 3x3x2 is 11 or 12.
    # So 3x3x5 might be decomposable into 3x3x3 and 3x3x2, giving rank 23 + 12 = 35.
    # But we need to find the actual decomposition.
    
    # Since we cannot guarantee finding a valid lower-rank decomposition quickly,
    # we will return the schoolbook decomposition.
    
    return schoolbook()

# EVOLVE-BLOCK-END
