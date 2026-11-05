"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on 2x2 sub-blocks to reduce rank below schoolbook 63.
    
    Decomposition:
    - Split C (3x7) into columns: [0:2], [2:4], [4:7]
    - For each column group, use appropriate sub-multiplication
    
    We'll use:
    - M_{2,3,2} with Strassen-like optimization for first two column groups
    - M_{2,3,3} for the last column group  
    - M_{1,3,7} for the last row of A
    
    Actually, let me use a cleaner approach:
    - Use 2x2 Strassen blocks where possible
    - The 3x3 A can be seen as having a 2x2 block plus extra row/col
    """
    n, m, p = 3, 3, 7
    
    # We'll build the decomposition term by term.
    # Each term is (U_row, V_row, W_row, coefficient)
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (21, rank)  
    # W has shape (p*n, rank) = (21, rank)
    
    # Indexing:
    # U index: i*m + j for A[i,j]
    # V index: j*p + k for B[j,k]
    # W index: k*n + i for C[i,k] (trace convention: W[k*n+i, r])
    
    terms = []  # Each term: (u_idx, v_idx, w_idx, coeff)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Strategy: Use Strassen on 2x2 sub-problems
    # 
    # Split:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    #
    # B = [[B11, B12, B13, B14, B15, B16, B17],
    #      [B21, B22, B23, B24, B25, B26, B27],
    #      [B31, B32, B33, B34, B35, B36, B37]]
    #
    # C = A * B, so C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's use a block decomposition:
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> 2x2x2, use Strassen (rank 7)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> 2x2x2, use Strassen (rank 7)  
    # Block 3: A[0:2, 0:2] * B[0:2, 4:7] -> 2x2x3, schoolbook rank 12
    # Block 4: A[0:2, 2] * B[2, 0:2] -> 2x1x2, rank 4
    # Block 5: A[0:2, 2] * B[2, 2:4] -> 2x1x2, rank 4
    # Block 6: A[0:2, 2] * B[2, 4:7] -> 2x1x3, rank 6
    # Block 7: A[2, 0:2] * B[0:2, 0:2] -> 1x2x2, rank 4
    # Block 8: A[2, 0:2] * B[0:2, 2:4] -> 1x2x2, rank 4
    # Block 9: A[2, 0:2] * B[0:2, 4:7] -> 1x2x3, rank 6
    # Block 10: A[2, 2] * B[2, 0:2] -> 1x1x2, rank 2
    # Block 11: A[2, 2] * B[2, 2:4] -> 1x1x2, rank 2
    # Block 12: A[2, 2] * B[2, 4:7] -> 1x1x3, rank 3
    # 
    # Total schoolbook: 7+7+12+4+4+6+4+4+6+2+2+3 = 61
    # But wait, this double-counts! The blocks overlap in the j-index.
    
    # Actually, the correct decomposition must partition the j-index (the shared dimension).
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # So we must sum over j=0,1,2. We can't split by j arbitrarily without ensuring
    # each (i,j,k) triple is covered exactly once.
    
    # The schoolbook decomposition has one term for each (i,j,k) in [0,3)x[0,3)x[0,7).
    # To use Strassen, we need to group terms cleverly.
    
    # Let me try a different approach: directly implement Strassen for the 2x2x2 
    # sub-problems embedded in the larger tensor.
    
    # For the 2x2x2 sub-problem with A indices (i0,i1), B indices (j0,j1), C indices (k0,k1):
    # The 7 Strassen terms are:
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j1,k0]+B[j1,k1])
    # M4 = (A[i0,j0]-A[i1,j0]) * (B[j1,k1]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    # M6 = A[i1,j1] * (B[j0,k0]-B[j1,k0])
    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j0,k0]+B[j1,k0])
    
    # And the C entries:
    # C[i0,k0] = M1 + M4 - M6 - M7
    # C[i0,k1] = M5 + M7
    # C[i1,k0] = M2 + M6
    # C[i1,k1] = M1 - M2 + M4
    
    # But in tensor decomposition form, each "M" is not a single rank-1 term.
    # The rank-1 terms are the atomic (u,v,w) triples.
    
    # Actually, in the tensor decomposition, the 7 terms of Strassen ARE the rank-1 terms.
    # Each term r has:
    # u_r: a vector in R^4 (the 2x2 A matrix flattened)
    # v_r: a vector in R^4 (the 2x2 B matrix flattened)
    # w_r: a vector in R^4 (the 2x2 C matrix flattened)
    
    # For M_{2,2,2}, the 7 rank-1 terms are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1)  -- NO, this is wrong
    
    # Let me look up the correct form. The Strassen decomposition gives:
    # T = sum_{r=1}^{7} u_r \otimes v_r \otimes w_r
    # where:
    # u_1 = (1, 0, 0, 1)   [A11+A22]
    # v_1 = (1, 0, 0, 1)   [B11+B22]
    # w_1 = (1, 0, 0, 1)   [C11+C22-C12-C21]? No.
    
    # I think I'm confusing myself. Let me use the explicit schoolbook for now
    # and try to find a valid reduction.
    
    # Fallback: schoolbook rank 63
    rank = n * m * p
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
