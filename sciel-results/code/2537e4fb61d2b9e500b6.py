import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    # 4 blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] (Rank 7)
    # 2. A[0:2, 2] * B[2, 0:2] (Rank 4)
    # 3. A[0:2, 0:2] * B[0:2, 2:4] (Rank 7)
    # 4. A[0:2, 2] * B[2, 2:4] (Rank 4)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # i_offset: 0, j_offset: 0, k_offset: 0 or 2
        # A indices: i in [i_offset, i_offset+2), j in [j_offset, j_offset+2)
        # B indices: j in [j_offset, j_offset+2), k in [k_offset, k_offset+2)
        # C indices: i in [i_offset, i_offset+2), k in [k_offset, k_offset+2)
        
        # M1 = A00 (B01 + B11)
        # U: A00 (i=0, j=0) -> 1
        # V: B01 (j=0, k=1), B11 (j=1, k=1) -> 1
        # W: C00 (k=0, i=0), C01 (k=1, i=0), C11 (k=1, i=1) ... 
        # Wait, the W vector accumulates contributions to C.
        # C00 += M1? No.
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M7
        # C10 = M3 + M5
        # C11 = M1 - M4 + M2 + M3
        
        # We need to map local indices to global.
        # Local A: (0,0), (0,1), (1,0), (1,1)
        # Local B: (0,0), (0,1), (1,0), (1,1)
        # Local C: (0,0), (0,1), (1,0), (1,1)
        
        # M1
        U[i_offset * m + j_offset][col_start] = 1 # A00
        V[j_offset * p + k_offset + 1][col_start] = 1 # B01
        V[(j_offset + 1) * p + k_offset + 1][col_start] = 1 # B11
        # W contributions: C01 (k=1, i=0) and C11 (k=1, i=1) get +1
        W[(k_offset + 1) * n + i_offset][col_start] = 1
        W[(k_offset + 1) * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        # M2 = (A00 + A10) B11
        U[i_offset * m + j_offset][col_start] = 1 # A00
        U[(i_offset + 1) * m + j_offset][col_start] = 1 # A10
        V[(j_offset + 1) * p + k_offset + 1][col_start] = 1 # B11
        # W contributions: C00 (k=0, i=0) -1, C11 (k=1, i=1) +1
        W[k_offset * n + i_offset][col_start] = -1
        W[(k_offset + 1) * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        # M3 = (A10 + A11) (B00 - B10)
        U[(i_offset + 1) * m + j_offset][col_start] = 1 # A10
        U[(i_offset + 1) * m + j_offset + 1][col_start] = 1 # A11
        V[j_offset * p + k_offset][col_start] = 1 # B00
        V[(j_offset + 1) * p + k_offset][col_start] = -1 # B10
        # W contributions: C10 (k=0, i=1) +1, C11 (k=1, i=1) +1
        W[k_offset * n + i_offset + 1][col_start] = 1
        W[(k_offset + 1) * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        # M4 = A11 (B00 + B10)
        U[(i_offset + 1) * m + j_offset + 1][col_start] = 1 # A11
        V[j_offset * p + k_offset][col_start] = 1 # B00
        V[(j_offset + 1) * p + k_offset][col_start] = 1 # B10
        # W contributions: C00 (k=0, i=0) +1, C11 (k=1, i=1) -1
        W[k_offset * n + i_offset][col_start] = 1
        W[(k_offset + 1) * n + i_offset + 1][col_start] = -1
        col_start += 1
        
        # M5 = (A00 + A11) (B00 + B11)
        U[i_offset * m + j_offset][col_start] = 1 # A00
        U[(i_offset + 1) * m + j_offset + 1][col_start] = 1 # A11
        V[j_offset * p + k_offset][col_start] = 1 # B00
        V[(j_offset + 1) * p + k_offset + 1][col_start] = 1 # B11
        # W contributions: C00 (k=0, i=0) +1, C10 (k=0, i=1) +1
        W[k_offset * n + i_offset][col_start] = 1
        W[k_offset * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        # M6 = (A00 - A10) B10
        U[i_offset * m + j_offset][col_start] = 1 # A00
        U[(i_offset + 1) * m + j_offset][col_start] = -1 # A10
        V[(j_offset + 1) * p + k_offset][col_start] = 1 # B10
        # W contributions: C00 (k=0, i=0) +1
        W[k_offset * n + i_offset][col_start] = 1
        col_start += 1
        
        # M7 = (A11 - A01) (B01 + B11)
        U[(i_offset + 1) * m + j_offset + 1][col_start] = 1 # A11
        U[i_offset * m + j_offset + 1][col_start] = -1 # A01
        V[j_offset * p + k_offset + 1][col_start] = 1 # B01
        V[(j_offset + 1) * p + k_offset + 1][col_start] = 1 # B11
        # W contributions: C01 (k=1, i=0) +1
        W[(k_offset + 1) * n + i_offset][col_start] = 1
        col_start += 1
        
        return col_start

    # Helper to add a 2x1x2 block (Schoolbook)
    def add_2x1x2_block(i_offset, j_index, k_offset, col_start):
        # A is 2x1 (rows i_offset, i_offset+1; col j_index)
        # B is 1x2 (row j_index; cols k_offset, k_offset+1)
        # C is 2x2 (rows i_offset, i_offset+1; cols k_offset, k_offset+1)
        
        # C00 = A0 B0
        U[i_offset * m + j_index][col_start] = 1
        V[j_index * p + k_offset][col_start] = 1
        W[k_offset * n + i_offset][col_start] = 1
        col_start += 1
        
        # C01 = A0 B1
        U[i_offset * m + j_index][col_start] = 1
        V[j_index * p + k_offset + 1][col_start] = 1
        W[(k_offset + 1) * n + i_offset][col_start] = 1
        col_start += 1
        
        # C10 = A1 B0
        U[(i_offset + 1) * m + j_index][col_start] = 1
        V[j_index * p + k_offset][col_start] = 1
        W[k_offset * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        # C11 = A1 B1
        U[(i_offset + 1) * m + j_index][col_start] = 1
        V[j_index * p + k_offset + 1][col_start] = 1
        W[(k_offset + 1) * n + i_offset + 1][col_start] = 1
        col_start += 1
        
        return col_start

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: A[0:2, 2] * B[2, 0:2]
    col = add_2x1x2_block(0, 2, 0, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4]
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 4: A[0:2, 2] * B[2, 2:4]
    col = add_2x1x2_block(0, 2, 2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
