"""Matrix multiplication 3x3x7 - rank-reduced tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 120  # safety margin
    
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks, combine for 3x3x3
    # 3x3x3 can be decomposed using overlapping 2x2 blocks with Strassen
    # Then handle the 3x3x1 part with schoolbook (rank 9)
    
    # For 3x3x3: use two overlapping 2x2x2 Strassen decompositions
    # Block 1: rows/cols 0,1 of A and B -> covers C[0:2,0:2]
    # Block 2: rows/cols 1,2 of A and B -> covers C[1:3,1:3]
    # Plus correction terms for C[0,2], C[2,0]
    
    # Actually, let me use a cleaner approach:
    # Split 3x3x7 into:
    # - 2x2x7 part (using Strassen for the 2x2x2 subblocks)
    # - 1x3x7 part (the remaining row of A)
    # - 3x1x7 part (the remaining column of A)
    # - 1x1x7 part (overlap correction)
    
    # Better: use block decomposition
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    # where B is 3x7
    
    # Use Strassen for the 2x2x2 blocks:
    # Top-left 2x2x2: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (rank 7)
    # Top-right 2x2x2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (rank 7)
    # etc.
    
    # Let me use a systematic approach with Strassen for 2x2 blocks
    
    def strassen_decomposition(n_block=2, m_block=2, p_block=2):
        """Return rank-7 decomposition for 2x2x2 matrix multiplication."""
        # Strassen's 7 products for 2x2x2
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        U = np.zeros((4, 7))  # 2x2 = 4 entries
        V = np.zeros((4, 7))  # 2x2 = 4 entries
        W = np.zeros((4, 7))  # 2x2 = 4 entries (output is 2x2)
        
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for 2x2 matrices
        # A = [[a00, a01], [a10, a11]] -> indices 0,1,2,3
        # B = [[b00, b01], [b10, b11]] -> indices 0,1,2,3
        # C = [[c00, c01], [c10, c11]] -> indices 0,1,2,3
        
        # M1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1  # a00, a11
        V[0, 0] = 1; V[3, 0] = 1  # b00, b11
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1  # all c's get M1
        
        # M2 = (a10+a11)b00
        U[2, 1] = 1; U[3, 1] = 1  # a10, a11
        V[0, 1] = 1  # b00
        W[2, 1] = 1; W[3, 1] = 1  # c10, c11
        
        # M3 = a00(b01-b11)
        U[0, 2] = 1  # a00
        V[1, 2] = 1; V[3, 2] = -1  # b01, -b11
        W[1, 2] = 1  # c01
        
        # M4 = a11(b10-b00)
        U[3, 3] = 1  # a11
        V[2, 3] = 1; V[0, 3] = -1  # b10, -b00
        W[2, 3] = 1  # c10
        
        # M5 = (a00+a01)b11
        U[0, 4] = 1; U[1, 4] = 1  # a00, a01
        V[3, 4] = 1  # b11
        W[0, 4] = 1; W[1, 4] = 1  # c00, c01
        
        # M6 = (a10-a00)(b00+b01)
        U[2, 5] = 1; U[0, 5] = -1  # a10, -a00
        V[0, 5] = 1; V[1, 5] = 1  # b00, b01
        W[2, 5] = 1; W[3, 5] = 1  # c10, c11
        
        # M7 = (a01-a11)(b10+b11)
        U[1, 6] = 1; U[3, 6] = -1  # a01, -a11
        V[2, 6] = 1; V[3, 6] = 1  # b10, b11
        W[1, 6] = 1; W[3, 6] = 1  # c01, c11
        
        return U, V, W
    
    # Build the full 3x3x7 decomposition
    # We'll use schoolbook for simplicity but try to optimize
    
    # Actually, let me use a hybrid: Strassen for 2x2x2 blocks + corrections
    # This is complex, so let me use a known lower-rank approach
    
    # For now, use schoolbook but with a cleaner structure
    rank = 63
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Schoolbook: C[i,k] = sum_j A[i,j] * B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
