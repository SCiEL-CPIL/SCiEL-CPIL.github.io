"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _try_rank(r, max_time=140.0):
    """Attempt to find a rank-r decomposition using algebraic methods."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # We want U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use the fact that the tensor is a sum of rank-1 tensors.
    # The standard decomposition has rank n*m*p.
    # We can try to find a lower rank decomposition by exploiting structure.
    
    # For 2x3x9, let's try to use a structured decomposition.
    # One approach: factor through a smaller intermediate space.
    
    # Let's try a decomposition based on Kronecker products or similar structures.
    
    # Actually, let's try a more direct algebraic approach.
    # We can represent the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For rank r, we need to find matrices U, V, W.
    # This is equivalent to finding a bilinear algorithm of rank r.
    
    # Let's try to use the Strassen-like approach or other known results.
    # For 2x3x9, the rank is at most 2*3*9 = 54.
    # Can we do better?
    
    # One known result: rank of (2,3,9) multiplication is at most 2*3*9 - 2*3 - 3*9 + 2 = 54 - 6 - 27 + 2 = 23? No, that's not a standard formula.
    
    # Let's try a constructive approach.
    # We can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    # This gives rank 3*9 = 27 for each row i, but we can share across rows.
    
    # Actually, the schoolbook is rank 2*3*9 = 54.
    
    # Let's try to find a rank-27 decomposition.
    # For each k, C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as a vector-matrix product.
    
    # Let's try a different approach: use the fact that we have only 2 rows in A.
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # We can group terms. Let's define new variables to reduce rank.
    
    # For now, let's return the schoolbook decomposition as a fallback.
    # But first, let's try a few specific ranks.
    
    # Try rank 27: one term for each (j,k) pair, shared across i.
    if r >= 27:
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        col = 0
        for j in range(m):
            for k in range(p):
                # We need to represent A[i,j]*B[j,k] for i=0,1
                # Using one rank-1 term: U[i*m+j] * V[j*p+k] * W[k*n+i]
                # But we need this for both i=0 and i=1 with the same (j,k).
                # This doesn't work directly with one term per (j,k).
                col += 1
        
        # This approach doesn't work. Let's try the block approach.
        # For each i, we have a 1x3 by 3x9 multiplication.
        # Rank for 1x3x9 is 3*9 = 27? No, it's 3*9 = 27 for schoolbook, but can be lower.
        # Actually, for 1xm by pxq, the rank is m*q? No.
        
        # Let's just return schoolbook for now.
        pass
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a rank-minimal decomposition.
    # For 2x3x9, let's try specific ranks.
    
    # The schoolbook rank is 54.
    # Can we do better?
    
    # Let's try a decomposition that exploits the small size.
    # One approach: use the fact that we have 2 rows in A.
    
    # For C[0,k] and C[1,k], we can write:
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # Let's define new products to reduce the number of terms.
    # For example, we can define:
    # D[0,k] = B[0,k], D[1,k] = B[1,k], D[2,k] = B[2,k]
    # Then C[0,k] = A[0,0]D[0,k] + A[0,1]D[1,k] + A[0,2]D[2,k]
    # C[1,k] = A[1,0]D[0,k] + A[1,1]D[1,k] + A[1,2]D[2,k]
    
    # This doesn't reduce the rank.
    
    # Let's try a different approach: use bilinear combinations.
    # For instance, define:
    # E[0] = A[0,0] + A[1,0], E[1] = A[0,0] - A[1,0]
    # F[0] = B[0,k] + B[0,k], F[1] = B[0,k] - B[0,k] (this doesn't help)
    
    # Actually, let's just return the schoolbook decomposition.
    # The rank is 54.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
