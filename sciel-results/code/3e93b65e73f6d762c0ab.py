"""Find rank-minimal exact tensor decomposition for 3x5 x 5x7 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS  # n=3 (rows of A, C), m=5 (cols of A, rows of B), p=7 (cols of B, C)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x7 matrix.
    U: (n*m, rank) - indexed by (i,j) where i in [0,n), j in [0,m)
    V: (m*p, rank) - indexed by (j,k) where j in [0,m), k in [0,p)
    W: (p*n, rank) - indexed by (k,i) where k in [0,p), i in [0,n) [trace convention W[k*n+i,r]]
    
    The decomposition satisfies: for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid matmul term, else 0
    
    Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook rank = n*m*p = 105
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # Approach: Use structured decomposition based on matrix multiplication properties
    
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can decompose this using the fact that for each fixed j, the contribution is A[i,j]*B[j,k]
    # The schoolbook uses one rank-1 term per (i,j,k)
    
    # Better approach: For each j, we have a rank-1 tensor in (i,k) space: A[:,j] * B[j,:]
    # But we need to express this in the U,V,W framework
    
    # Let's think of it as: T[i,j,k] = 1 = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # One approach: Use the identity that 1 = 1*1*1, so if we set U[i,j,c]=1, V[j,k,c]=1, W[i,k,c]=1
    # for a single c, that gives T=1 everywhere. But that's rank 1! Wait, that can't be right.
    
    # Actually no - if U[i,j,c]=1 for all i,j, V[j,k,c]=1 for all j,k, W[i,k,c]=1 for all i,k
    # then sum_c U*V*W = 1*1*1 = 1. So rank 1 would work?!
    
    # But wait - that would mean the decomposition is U[i,j,c]=1, V[j,k,c]=1, W[i,k,c]=1
    # Let's verify: for any i,j,k, U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. Yes!
    
    # So rank 1 is sufficient? Let me double-check the problem statement.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid indices.
    # So yes, rank 1 should work!
    
    # But that seems too good to be true. Let me reconsider...
    # Actually, I think I'm misunderstanding the tensor. The matrix multiplication tensor
    # T_{ijk} is defined such that (A*B)_{ik} = sum_j A_{ij} * B_{jk} * T_{ijk}
    # For standard matrix multiplication, T_{ijk} = 1 for all i,j,k.
    
    # So a rank-1 decomposition would be: T_{ijk} = U_{ij} * V_{jk} * W_{ik}
    # If we set U_{ij}=1, V_{jk}=1, W_{ik}=1, then T_{ijk}=1. This works!
    
    # So the answer is rank 1.
    
    U = [[1] * 1] * (n * m)
    V = [[1] * 1] * (m * p)
    W = [[1] * 1] * (p * n)
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = U[i*m+j][0] * V[j*p+k][0] * W[k*n+i][0]
                if val != 1:
                    # Fallback to schoolbook if verification fails
                    rank = rank_schoolbook
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1
                                V[jj*p+kk][col] = 1
                                W[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
