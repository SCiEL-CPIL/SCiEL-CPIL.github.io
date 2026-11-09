"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _block_kronecker_decomposition(n, m, p):
    """
    Use block-wise Kronecker product decomposition.
    Treat n x m matrix as block row of 1x5 vectors (n blocks)
    Treat m x p matrix as block column of 1x6 vectors (m blocks)
    
    The rank of multiplying a 1x5 vector by a 5x6 matrix is 5 (schoolbook for that sub-problem).
    So the total rank is n * 5 = 15.
    
    Structure:
    For each i in range(n):
        For each c in range(5):
            This component handles the contribution from row i of A and column c of B
            U[i*m + j, i*5 + c] = 1 if j == c (since A is 1x5, row i has entries at positions 0..4)
            Wait, let me think more carefully.
            
    Actually, let's think of it differently:
    A is 3x5, B is 5x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we use rank 15, we can have components indexed by (i, j) where i in 0..2, j in 0..4:
    U[i*m + j, i*5 + j] = 1  (A[i,j] picks out the j-th element of row i)
    V[j*p + k, i*5 + j] = 1  (B[j,k] picks out the k-th element of column j)
    W[k*n + i, i*5 + j] = 1  (contributes to C[i,k])
    
    Let's verify: For a given (i, k), the sum over all components (i', j) is:
    sum_{i', j} U[i*m+j, i'*5+j] * V[j*p+k, i'*5+j] * W[k*n+i, i'*5+j]
    
    U[i*m+j, i'*5+j] = 1 iff i' == i (since the row index is i*m+j, and column is i'*5+j, 
    so they match only if i' == i)
    Wait, that's not right. U has shape (n*m, rank) = (15, 15).
    U[row, col] where row = i*m + j, col = i'*5 + j'
    
    Let me re-derive. We want rank = n * m = 15.
    Component index: (i, j) for i in 0..n-1, j in 0..m-1.
    col = i * m + j (range 0..14)
    
    U[i*m + j, col] = 1  (this picks out A[i,j])
    V[j*p + k, col] = 1  (this picks out B[j,k])
    W[k*n + i, col] = 1  (this puts the product into C[i,k])
    
    Verification: For fixed (i, k), sum over all (i', j'):
    sum_{i', j'} U[i*m + j', i'*m + j'] * V[j'*p + k, i'*m + j'] * W[k*n + i, i'*m + j']
    
    U[i*m + j', i'*m + j'] = 1 iff i == i' and j' == j' (always true for the row/col match)
    Actually, U[row, col] = 1 iff row == i*m + j and col == i*m + j for some (i,j).
    So U[i*m + j', col] = 1 iff col == i*m + j'
    
    V[j'*p + k, col] = 1 iff col == j'*p + k ... no wait.
    V[row, col] = 1 iff row == j*p + k and col == i*m + j for some (i,j).
    So V[j'*p + k, col] = 1 iff col == i*m + j where the j in col must equal j' for the row to match.
    Actually, V[row, col]: row = j*p + k, col = i*m + j.
    For a fixed k and j', V[j'*p + k, col] = 1 iff col == i*m + j' for some i.
    
    W[k*n + i, col] = 1 iff col == i*m + j for some j.
    
    So the product U[i*m+j', col] * V[j'*p+k, col] * W[k*n+i, col] is non-zero only when:
    - col == i*m + j' (from U)
    - col == i'*m + j' for some i' (from V, but since row is j'*p+k, the j in col must be j')
    - col == i*m + j'' for some j'' (from W, but since row is k*n+i, the i in col must be i)
    
    So all three require col == i*m + j'. Then:
    U[i*m+j', i*m+j'] = 1
    V[j'*p+k, i*m+j'] = 1
    W[k*n+i, i*m+j'] = 1
    
    Sum over j': sum_{j'=0}^{m-1} 1*1*1 = m = 5
    
    But we need the sum to be 1, not 5! So we need to scale by 1/5.
    
    Let me adjust: U[i*m+j, col] = 1/5, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    Then sum = (1/5) * 1 * 1 * 5 = 1. Correct!
    
    Or: U = 1, V = 1, W = 1/5. Or distribute the scaling.
    """
    n, m, p = 3, 5, 6
    rank = n * m  # 15
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            col = i * m + j
            U[i * m + j, col] = 1.0
            V[j * p, col] = 1.0  # Wait, V row should be j*p + k, but k varies
            # Actually V[j*p + k, col] = 1 for all k
            for k in range(p):
                V[j * p + k, col] = 1.0
            W[:, col] = 0  # Will set below
            for k in range(p):
                W[k * n + i, col] = 1.0 / m  # Scale by 1/m to get sum = 1
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_r(n, m, p, r, max_time=10):
    """Try to find a rank-r decomposition using structured Kronecker approach."""
    start_time = time.time()
    
    # First try the block Kronecker approach if r == n*m
    if r == n * m:
        result = _block_kronecker_decomposition(n, m, p)
        if result is not None:
            return result
    
    # For other ranks, fall back to schoolbook or return None
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try the block Kronecker decomposition first (rank = n*m = 15)
    result = _block_kronecker_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
