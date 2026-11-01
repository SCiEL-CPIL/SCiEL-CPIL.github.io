"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # Known minimal rank for (2,2,p) matrix multiplication is 4p-2 for p >= 3.
    # For p=9, rank = 34.
    # 
    # Construction strategy:
    # We need to compute C = A * B where A is 2x2, B is 2x9, C is 2x9.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using the trace convention: W[k*n+i, r] corresponds to C[k,i] but our C is 2x9
    # with C[i,k]. Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So C has dimensions p x n = 9 x 2, with C[k,i].
    # 
    # Wait, the task says "2 by 2 matrix by a 2 by 9 matrix". So A is 2x2, B is 2x9.
    # The result C should be 2x9. But the trace convention says W[k*n+i,r], which suggests
    # C is indexed as C[k,i] where k is 0..p-1=0..8 and i is 0..n-1=0..1.
    # This means C is actually p x n = 9 x 2 in the W factor, but logically it's the same
    # as a 2x9 matrix transposed, or the indexing is just different.
    # 
    # Let me interpret: the multiplication tensor has indices (i, j, k) where:
    # - i: row of A (0..n-1=0..1)
    # - j: column of A / row of B (0..m-1=0..1)  
    # - k: column of B (0..p-1=0..8)
    # 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the factor matrices:
    # - U[i*m+j, r] = U[i*2+j, r] for A factor
    # - V[j*p+k, r] = V[j*9+k, r] for B factor
    # - W[k*n+i, r] = W[k*2+i, r] for C factor
    # 
    # So C[k,i] in the W factor corresponds to C[i,k] in standard notation.
    # 
    # Rank-34 construction for (2,2,p):
    # For each k, we have:
    #   C[0,k] = a*B[0,k] + b*B[1,k]
    #   C[1,k] = c*B[0,k] + d*B[1,k]
    # 
    # Standard schoolbook uses 4p terms. To reduce to 4p-2, we use:
    # For k=0..p-1:
    #   Term 1: a * B[0,k] -> C[0,k]
    #   Term 2: c * B[0,k] -> C[1,k]
    # For k=0..p-1:
    #   Term 3: (b+d) * B[1,k] -> C[0,k] and C[1,k]
    # For k=0..p-2:
    #   Term 4: d * (B[1,k] - B[0,k]) -> adjust C[0,k] and C[1,k]
    # 
    # Actually, let me use a cleaner known construction.
    # 
    # The standard rank 4p-2 construction:
    # For k in 0..p-1:
    #   r1: A[0,0] * B[0,k] -> C[0,k]
    #   r2: A[1,0] * B[0,k] -> C[1,k]
    # For k in 0..p-1:
    #   r3: (A[0,1]+A[1,1]) * B[1,k] -> C[0,k] + C[1,k]
    # For k in 0..p-2:
    #   r4: A[1,1] * (B[1,k] - B[0,k]) -> C[1,k] - C[0,k] adjustment
    # 
    # Let me verify: 
    # C[0,k] = A[0,0]*B[0,k] + (A[0,1]+A[1,1])*B[1,k] - A[1,1]*(B[1,k]-B[0,k])
    #        = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[1,1]*B[1,k] - A[1,1]*B[1,k] + A[1,1]*B[0,k]
    #        = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[1,1]*B[0,k]
    # That's wrong. Let me try a different approach.
    
    # Alternative: use 2p terms for the first column of B, and 2p-2 terms for the second.
    # 
    # For k=0..p-1:
    #   t1: A[0,0]*B[0,k] -> C[0,k]
    #   t2: A[1,0]*B[0,k] -> C[1,k]
    # For k=0..p-1:
    #   t3: A[0,1]*B[1,k] -> C[0,k]
    # For k=0..p-2:
    #   t4: A[1,1]*B[1,k] -> C[1,k]
    # Plus for k=p-1:
    #   t5: A[1,1]*B[1,p-1] -> C[1,p-1]
    # That's 2p + p + (p-1) + 1 = 4p. Not reduced.
    
    # Let me look up the actual construction. The minimal rank for (2,2,p) is indeed 4p-2.
    # 
    # Construction:
    # For k=0..p-1:
    #   s1: A[0,0]*B[0,k] -> C[0,k]
    #   s2: A[1,0]*B[0,k] -> C[1,k]
    # For k=0..p-1:
    #   s3: (A[0,1]+A[1,1])*B[1,k] -> C[0,k] + C[1,k]
    # For k=0..p-2:
    #   s4: A[1,1]*(B[1,k]-B[0,k]) -> C[1,k] - C[0,k]
    # 
    # Check C[0,k]: A[0,0]*B[0,k] + (A[0,1]+A[1,1])*B[1,k]
    # Check C[1,k]: A[1,0]*B[0,k] + (A[0,1]+A[1,1])*B[1,k] - A[1,1]*(B[1,k]-B[0,k])
    #              = A[1,0]*B[0,k] + A[0,1]*B[1,k] + A[1,1]*B[1,k] - A[1,1]*B[1,k] + A[1,1]*B[0,k]
    #              = A[1,0]*B[0,k] + A[0,1]*B[1,k] + A[1,1]*B[0,k]
    # That's still wrong for C[1,k]. We want C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k].
    
    # I think I need to be more careful. Let me just implement the schoolbook with rank 36
    # and verify it's correct, since the rank-34 construction is tricky.
    
    rank = 36
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((18, rank), dtype=float)
    W = np.zeros((18, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
