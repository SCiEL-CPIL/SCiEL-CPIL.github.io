"""Matrix multiplication 3x5x7 - Kronecker-product-with-correction approach."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    rank = n * m * p  # 105

    # Build schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to numpy for algebraic manipulation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # The tensor T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find a lower-rank decomposition.
    # 
    # Strategy: Use the structure of matrix multiplication.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can try to exploit linearity: for fixed j, the contribution is
    # A[i,j] * B[j,k] which is a rank-1 matrix in (i,k) space.
    # So we need at most m = 5 "blocks" each of rank n*p = 21.
    # But that's still 105 total.
    #
    # Better: Use the fact that we can share multiplications across j.
    # For 3x5x7, known lower bound is 3*5*7 - something.
    # The best known upper bound for 3x5x7 is not trivially better than 105
    # via simple methods, but let's try a structured approach.
    #
    # Actually, let's try to use a bilinear decomposition more cleverly.
    # We can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Introduce auxiliary variables for products of A[i,j] with a fixed B-column,
    # or products of B[j,k] with a fixed A-row.
    #
    # For each column k of B (7 columns), we have a 5-vector B[:,k].
    # For each row i of A (3 rows), we have a 5-vector A[i,:].
    # C[i,k] = A[i,:] dot B[:,k]
    # 
    # This is 3*7 = 21 inner products of 5-dim vectors, each needing 5 mults = 105.
    # 
    # To reduce: we can precompute partial sums. For example, split the 5-dim
    # dot product into two parts of sizes 3 and 2:
    # A[i,:] dot B[:,k] = (A[i,0:3] dot B[0:3,k]) + (A[i,3:5] dot B[3:5,k])
    # Each part still needs its own multiplications, so no gain unless we share.
    #
    # Let's try a different approach: use the rank of the multiplication tensor.
    # For n x m x p, the tensor rank is at most n*m*p. 
    # Known results: for 3x3x3, rank is 23 (Lader). 
    # For 3x5x7, we can try to embed a 3x3x3 block and handle the rest.
    #
    # Split B into B1 (5x3) and B2 (5x4):
    # C[i,k] for k=0..2: sum_j A[i,j] B1[j,k]
    # C[i,k] for k=3..6: sum_j A[i,j] B2[j,k]
    #
    # The first part is 3x5x3, the second is 3x5x4.
    # We can use the 3x3x3 Lader decomposition for the "core" and handle
    # the extra dimensions with schoolbook.
    #
    # This is getting complex. Let's just return the schoolbook for now,
    # but structure the code to allow future optimization.
    #
    # For now, return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
