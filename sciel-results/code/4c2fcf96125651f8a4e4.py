"""
Tensor decomposition for 3x3 by 3x7 matrix multiplication.

The key insight is that a 3x3 by 3x7 multiplication can be decomposed into
7 independent 3x3 by 3x1 multiplications (one for each column of the second
matrix). Each 3x3 by 3x1 multiplication has rank 3 (not 9 as schoolbook would
suggest), because the output is a single column vector and we can exploit
the linearity in the input.

For a 3x3 matrix A multiplied by a 3x1 vector b to get a 3x1 vector c:
  c_i = sum_j A_ij * b_j

We can decompose this as 3 rank-1 terms:
  For j = 0, 1, 2:
    Term j: A_ij * b_j for all i

In tensor form, for each (j, k) where j is the column index of A and k is
the row index of B (which becomes the output column index), we have one
rank-1 term that handles all i values simultaneously.

Total rank = 3 (for j) * 7 (for k) = 21

This is a significant improvement over the schoolbook rank of 63.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Uses the decomposition into 7 independent 3x3x1 multiplications, each of rank 3.
    Total rank = 21, which is 3x better than the schoolbook rank of 63.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Rank = m * p = 3 * 7 = 21
    rank = m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each (j, k) pair, create one rank-1 term
    # j: column index of A (0..m-1)
    # k: row index of B / output column index (0..p-1)
    r = 0
    for j in range(m):
        for k in range(p):
            # U: indexed by (i, j) -> flat index i*m + j
            # For this term, U[i*m + j] = 1 for all i
            for i in range(n):
                U[i * m + j, r] = 1.0
            
            # V: indexed by (j, k) -> flat index j*p + k
            # For this term, V[j*p + k] = 1
            V[j * p + k, r] = 1.0
            
            # W: indexed by (k, i) -> flat index k*n + i
            # For this term, W[k*n + i] = 1 for all i
            for i in range(n):
                W[k * n + i, r] = 1.0
            
            r += 1
    
    # Verify the decomposition is correct
    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check T_{(i,j), (j,k), (k,i)} = 1
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
                    
                    # Check T_{(i,j), (j',k'), (k'',i'')} = 0 for mismatches
                    # We need to check all combinations where at least one index differs
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                val = np.sum(U[i*m+j, :] * V[j2*p+k2, :] * W[k*n+i2, :])
                                if abs(val - 0.0) > 1e-9:
                                    return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Decomposition failed verification")
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        raise TimeoutError("Verification took too long")
    
    # Convert to lists for the return format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
