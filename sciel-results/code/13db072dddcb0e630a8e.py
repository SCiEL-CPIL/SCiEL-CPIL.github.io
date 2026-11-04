"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    # Trivial rank is n*m*p = 50
    # We can do better using Strassen-like block decompositions.
    # For 2x5 * 5x5, we can split the 5x5 into blocks.
    # Let's try a block decomposition approach.
    
    deadline = time.time() + 140
    
    # Strategy: Use block matrix multiplication.
    # A is 2x5, B is 5x5, C is 2x5.
    # Split B into horizontal blocks: B = [B1 | B2] where B1 is 5x2, B2 is 5x3.
    # Then C = A1*B1 + A2*B2 where A1 = A[:, 0:2] (2x2), A2 = A[:, 2:5] (2x3).
    # This doesn't directly help unless we use Strassen on the blocks.
    
    # Better approach: Use the fact that we can multiply 2x5 * 5x5 by:
    # Splitting into 2x2 * 2x2 blocks where possible.
    # Let's try: A = [A1 A2] where A1 is 2x2, A2 is 2x3
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = A1*B11 + A2*B21 (2x2) + A1*B12 + A2*B22 (2x3)
    # This is just standard multiplication.
    
    # Let's try a different approach: use rank minimization via random sampling.
    # The tensor T has entries T[i,j,k] = 1 if the multiplication A*B produces C[i,k] with A[i,j]*B[j,k].
    # Actually, the multiplication tensor M_{i,j,k} = 1 for all i,j,k (it's the identity tensor for matrix mult).
    # Wait, no. The matrix multiplication tensor for n x m * m x p has rank at least n*m*p in the trivial case.
    # But we can do better.
    
    # Let's use a known result: the rank of matrix multiplication <n,m,p> is bounded by various results.
    # For 2x5*5x5, we can try to find a decomposition.
    
    # Approach: Use random linear combinations to find a low-rank decomposition.
    # We want to find rank r and matrices U (10 x r), V (25 x r), W (10 x r) such that:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = A[i,j] * B[j,k] for all i,j,k
    
    # This is equivalent to: the tensor T with T[i,j,k] = e_{i,j} * e_{j,k} * e_{k,i} (in some sense)
    # Actually, the matrix multiplication tensor is:
    # M = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # No, that's not right either.
    
    # The matrix multiplication tensor <n,m,p> is defined as:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # Wait, no. The standard definition is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector for the (i,j) entry.
    
    # So the tensor has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # The rank is the minimum r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # For our case, n=2, m=5, p=5.
    # The trivial rank is 2*5*5 = 50.
    
    # Let's try to find a better decomposition using random methods.
    # We'll use the fact that the tensor is sparse and structured.
    
    # Actually, let's just use the trivial decomposition for now, since finding a better one
    # requires sophisticated algorithms that may not be feasible in the time limit.
    
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
