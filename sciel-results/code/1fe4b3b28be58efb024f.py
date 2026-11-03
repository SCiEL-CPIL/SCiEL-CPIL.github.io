"""Matrix multiplication 4,4,6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices representing the tensor decomposition.
    Uses the schoolbook decomposition which is guaranteed to be exact and valid.
    Rank = n * m * p = 4 * 4 * 6 = 96.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # The tensor T has entries T[i, j, k] = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    # Wait, let me re-read the convention.
    # The problem states: "The output factor follows the official trace convention W[k*n+i,r]."
    # This suggests the output C is of size p x n? No, that's unusual.
    # Let's look at the interface:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Standard matrix multiplication: C = A * B
    # A is n x m
    # B is m x p
    # C is n x p
    
    # However, the shapes in the interface are:
    # U: n*m rows (flattened A indices)
    # V: m*p rows (flattened B indices)
    # W: p*n rows (flattened C indices, but in what order?)
    
    # The convention "W[k*n+i,r]" where i is row of C and k is col of C?
    # If C is n x p, then i ranges 0..n-1 and k ranges 0..p-1.
    # Then the index would be i*p + k. But the interface says p*n rows and W[k*n+i].
    # This implies the first index k ranges 0..p-1 and the second index i ranges 0..n-1.
    # So W is indexed by (k, i) where k is the column of C and i is the row of C?
    # Or is C transposed?
    
    # Let's check the tensor definition for matrix multiplication.
    # Usually, the tensor T[i, j, k] for multiplying A (i x j) by B (j x k) gives C (i x k).
    # Here n=4, m=4, p=6.
    # A is 4x4, B is 4x6, C is 4x6.
    # So i in 0..3, j in 0..3, k in 0..5.
    # The tensor T has non-zero entries T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] contributes to C[i,k].
    # This is always true for the schoolbook algorithm.
    
    # Now, how are U, V, W defined?
    # U has n*m = 16 rows. These correspond to the entries of A. Index u in 0..15.
    # If A is flattened row-major, u = i*m + j = i*4 + j.
    # V has m*p = 24 rows. These correspond to the entries of B. Index v in 0..23.
    # If B is flattened row-major, v = j*p + k = j*6 + k.
    # W has p*n = 24 rows. These correspond to the entries of C.
    # The convention is W[k*n + i].
    # Here k is the column index of C (0..5) and i is the row index of C (0..3).
    # So the index w = k*n + i = k*4 + i.
    # This means C is stored in column-major order? Or just that the index is k first then i.
    
    # So for a rank-1 term r:
    # U[u, r] * V[v, r] * W[w, r] should sum to T[i, j, k] for the corresponding u, v, w.
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # Let's verify the schoolbook decomposition.
    # Rank = n * m * p = 4 * 4 * 6 = 96.
    # For each (i, j, k), we have one rank-1 term.
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
