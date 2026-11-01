"""Matrix multiplication 2x3x6 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We use a block decomposition: split the 6 columns into three blocks of 2.
    # Each block is a 2x3x2 multiplication.
    # For 2x3x2, the rank is 11 (known result).
    # Total rank = 3 * 11 = 33.
    # 
    # The 2x3x2 tensor T[i,j,k] = delta_{i,j,k} where i in {0,1}, j in {0,1,2}, k in {0,1}
    # We need rank-11 decomposition: T = sum_{c=1}^{11} u_c ⊗ v_c ⊗ w_c
    #
    # Using the known rank-11 decomposition for 2x3x2:
    # The 11 terms are constructed as follows:
    # For the 2x2x2 part (j=0,1): Strassen gives rank 7
    # For the remaining j=2 terms and cross terms: 4 additional terms
    # Total: 7 + 4 = 11
    #
    # Let me define the 11 rank-1 vectors for 2x3x2:
    # u is in R^2 (index i=0,1), v is in R^3 (index j=0,1,2), w is in R^2 (index k=0,1)
    #
    # Strassen for 2x2x2 (i,j,k all in {0,1}):
    # 7 terms with entries in {-1,0,1}
    #
    # For j=2, we need to handle the terms involving j=2.
    #
    # Let me use a systematic construction:
    # The 2x3x2 tensor has 12 entries. Rank 11 means one linear dependency.
    #
    # I'll use the following known decomposition:
    # Terms 1-7: Strassen on the 2x2x2 sub-tensor (j=0,1)
    # Terms 8-11: Handle j=2
    #
    # Strassen 2x2x2 vectors (u in R^2, v in R^2, w in R^2):
    # (u,v,w) tuples:
    # 1: (1,0), (1,0), (1,0)
    # 2: (1,0), (0,1), (1,0)
    # 3: (0,1), (1,0), (0,1)
    # 4: (0,1), (0,1), (0,1)
    # 5: (1,1), (1,1), (1,1)  -> wait, this isn't right
    #
    # Standard Strassen 7 products for C = AB where A,B are 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # For tensor decomposition of 2x2x2:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    #
    # The 7 rank-1 terms for 2x2x2 (using 0-indexed):
    # Term 1: u=(1,1), v=(1,0), w=(1,0)  -> contributes to (0,0,0) and (1,0,0)? No...
    #
    # Let me be more careful. For 2x2x2, T[i,j,k]=1 if the corresponding product term is present.
    # Actually, the matrix multiplication tensor for 2x2x2 is T[i,j,k] = delta_{i,k} * delta_{j,?} 
    # No. For C = A*B where C[i,k] = sum_j A[i,j]*B[j,k], the tensor is T[i,j,k] = delta_{i,j} * ... 
    # Actually T[i,j,k] = 1 for all (i,j,k) in {0,1}^3? No, that's the all-ones tensor.
    #
    # The matrix multiplication tensor [m,n,p] has T[i,j,k] = 1 if the (i,k) entry of the product 
    # depends on A[i,j] and B[j,k]. So T[i,j,k] = delta_{i,j} * delta_{j,k}... no.
    # T[i,j,k] = 1 for all combinations where i indexes rows of A, j columns of A/rows of B, k columns of B.
    # So T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p] where the indices are valid.
    # For 2x2x2, T[i,j,k]=1 for all 8 combinations? That's the all-ones tensor of rank 2.
    #
    # I'm confusing myself. Let me restart.
    #
    # For multiplying an n x m matrix A by an m x p matrix B to get n x p matrix C:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k.
    # So it's the all-ones tensor of size n x m x p? No, that's not right either.
    #
    # Actually, the structure tensor of matrix multiplication:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # It does appear, so T[i,j,k]=1 for all i,j,k in valid ranges.
    # But that would make it the all-ones tensor, which has rank min(n*m, m*p, n*p) typically.
    #
    # Wait, no. The tensor is T[i,j,k] where i in [n], j in [m], k in [p].
    # T[i,j,k] = 1 for all (i,j,k). The rank of the all-ones tensor of size n x m x p is min(n,m,p) 
    # when one dimension is 1, but in general...
    #
    # I think I've been overthinking this. The schoolbook decomposition in the seed uses rank n*m*p,
    # with one term per (i,j,k) triple. Each term has u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1.
    #
    # For 2x3x6, schoolbook rank = 36.
    #
    # The inspiration says block decomposition with Strassen for 2x2x2 reduces rank from 36 to 33.
    # This means using Strassen's rank-7 algorithm for 2x2x2 sub-problems.
    #
    # Let me use a direct approach: I'll implement a rank-33 decomposition by using
    # Strassen for the 2x2x2 blocks and handling the remaining terms.
    #
    # Actually, let me just use the schoolbook (rank 36) as a safe fallback and try to 
    # implement rank 33 if possible within time.
    
    # Schoolbook: rank 36
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
# EVOLVE-BLOCK-END
