import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # A is 2x5, split into A1 (2x2), A2 (2x2), A3 (2x1)
    # B is 5x6, split into B1 (2x6), B2 (2x6), B3 (1x6)
    # C = A1 B1 + A2 B2 + A3 B3
    
    # Rank contributions:
    # A1 B1: 2x2 x 2x6. Split B1 into three 2x2 blocks. 3 * Rank(2x2x2) = 3 * 7 = 21
    # A2 B2: 2x2 x 2x6. Split B2 into three 2x2 blocks. 3 * Rank(2x2x2) = 3 * 7 = 21
    # A3 B3: 2x1 x 1x6. 6 multiplications.
    # Total Rank = 21 + 21 + 6 = 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # A_local is 2x2, B_local is 2x2, C_local is 2x2
    # Indices in global A: i is i_offset, j is j_offset..j_offset+1
    # Indices in global B: j is j_offset..j_offset+1, k is k_offset..k_offset+1
    # Indices in global C: i is i_offset, k is k_offset..k_offset+1
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        
        # Local indices for A: (0,0), (0,1), (1,0), (1,1)
        # Local indices for B: (0,0), (0,1), (1,0), (1,1)
        # Local indices for C: (0,0), (0,1), (1,0), (1,1)
        
        # Global index calculation helpers
        def u_idx(li, lj):
            return i_offset * m + (j_offset + lj)
            
        def v_idx(lj, lk):
            return (j_offset + lj) * p + (k_offset + lk)
            
        def w_idx(lk, li):
            return (k_offset + lk) * n + i_offset
            
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += 1  # C00
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1
        
        # Term 2: M2 = (A10 + A11)B00
        U[u_idx(1, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        W[w_idx(0, 1)][col] += 1  # C10
        W[w_idx(1, 1)][col] += -1 # C11
        col += 1
        
        # Term 3: M3 = A00(B01 - B11)
        U[u_idx(0, 0)][col] += 1
        V[v_idx(0, 1)][col] += 1
        V[v_idx(1, 1)][col] += -1
        W[w_idx(1, 0)][col] += 1  # C01
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1
        
        # Term 4: M4 = A11(B10 - B00)
        U[u_idx(1, 1)][col] += 1
        V[v_idx(1, 0)][col] += 1
        V[v_idx(0, 0)][col] += -1
        W[w_idx(0, 0)][col] += 1  # C00
        W[w_idx(0, 1)][col] += 1  # C10
        col += 1
        
        # Term 5: M5 = (A00 + A01)B11
        U[u_idx(0, 0)][col] += 1
        U[u_idx(0, 1)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += -1 # C00
        W[w_idx(1, 0)][col] += 1  # C01
        col += 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        U[u_idx(1, 0)][col] += 1
        U[u_idx(0, 0)][col] += -1
        V[v_idx(0, 0)][col] += 1
        V[v_idx(0, 1)][col] += 1
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        U[u_idx(0, 1)][col] += 1
        U[u_idx(1, 1)][col] += -1
        V[v_idx(1, 0)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += 1  # C00
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1

    # Block 1: A1 (rows 0-1 of A, cols 0-1 of A) * B1 (rows 0-1 of B, cols 0-5 of B)
    # B1 is 2x6. Split into B1_0 (cols 0-1), B1_1 (cols 2-3), B1_2 (cols 4-5)
    add_strassen(0, 0, 0) # A1 * B1_0
    add_strassen(0, 0, 2) # A1 * B1_1
    add_strassen(0, 0, 4) # A1 * B1_2
    
    # Block 2: A2 (rows 0-1 of A, cols 2-3 of A) * B2 (rows 2-3 of B, cols 0-5 of B)
    # A2 is at i=0,1 and j=2,3. So i_offset=0, j_offset=2.
    # B2 is at j=2,3 and k=0..5.
    add_strassen(0, 2, 0) # A2 * B2_0
    add_strassen(0, 2, 2) # A2 * B2_1
    add_strassen(0, 2, 4) # A2 * B2_2
    
    # Block 3: A3 (rows 0-1 of A, col 4 of A) * B3 (row 4 of B, cols 0-5 of B)
    # A3 is 2x1. B3 is 1x6.
    # C3 = A3 * B3.
    # C3[i, k] = A3[i, 0] * B3[0, k]
    # A3[i, 0] is A[i, 4]. So U[i * m + 4].
    # B3[0, k] is B[4, k]. So V[4 * p + k].
    # C[i, k] is W[k * n + i].
    for k in range(6):
        U[0 * m + 4][col] = 1
        U[1 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
