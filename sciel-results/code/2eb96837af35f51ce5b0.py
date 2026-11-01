"""Program for rank-minimal exact tensor decomposition of 2x2 * 2x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We treat the multiplication as 8 independent 2x2 matrix-vector products.
    # For each column k of B (0..7), we compute C[:,k] = A @ B[:,k].
    # We use a rank-7 decomposition for each 2x2x2 tensor.
    # The total rank will be 8 * 7 = 56.
    
    # Known rank-7 decomposition for 2x2x2 matrix multiplication.
    # This is a standard result (Strassen-like or exact rank-7).
    # We need U0, V0, W0 of shape (4,7), (2,7), (2,7) such that
    # for each k, the k-th column of the result is computed by:
    # C0_k = sum_r W0[r] * (sum_{ij} U0[i,j,r] A[i,j]) * (sum_j V0[j,r] B[j,k])
    # Wait, the indices for V are (m*p, rank). For a single column k, V is (m, 1) effectively.
    # Let's construct the full matrices.
    
    # We will define the rank-7 decomposition coefficients for a single 2x2x2 block.
    # Let the decomposition be:
    # C_{i,k} = sum_{r=1}^7 w_{i,k,r} * (sum_{j} a_{i,j,r} * A_{i,j}) * (sum_{l} b_{j,l,r} * B_{j,l})
    # Actually, the standard form is:
    # C_{i,k} = sum_r W_{k*n+i, r} * M_r
    # M_r = (sum_{i,j} U_{i*m+j, r} A_{i,j}) * (sum_{j,k} V_{j*p+k, r} B_{j,k})
    
    # For the 2x2x2 case (n=2, m=2, p=2), we need U0 (4x7), V0 (4x7), W0 (4x7).
    # Let's use the known rank-7 decomposition.
    # One such decomposition uses the following pattern (from literature on tensor rank of 2x2x2):
    
    # We can use the decomposition from:
    # "The rank of the 2x2x2 matrix multiplication tensor is 7"
    # A common construction uses the following 7 terms:
    
    # We will use the schoolbook decomposition of rank 4 for each block, giving total rank 32.
    # This is safe and correct.
    
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
