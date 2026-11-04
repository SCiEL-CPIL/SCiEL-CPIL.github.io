import numpy as np

def construct():
    n, m, p = 2, 4, 5
    
    # We will build the decomposition using blocks.
    # Block 1: A[:, 0:2] * B[0:2, :]
    # Block 2: A[:, 2:4] * B[2:4, :]
    
    # Each block is 2x2x5.
    # 2x2x5 is decomposed into 2x2x2 (rank 7) + 2x2x3 (rank 10)
    # 2x2x3 is decomposed into 2x2x2 (rank 7) + 2x2x1 (rank 4)
    # So 2x2x5 is 2x2x2 + 2x2x2 + 2x2x1. Rank 7+7+4=18.
    
    # Total rank = 18 + 18 = 36.
    
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen block
    # offset_u, offset_v, offset_w are indices into U, V, W
    # j_offset is the column offset in A and row offset in B (0 or 2)
    # k_offset is the column offset in B and W (0, 2, or 4)
    
    def add_strassen_block(j_offset, k_offset, col_offset):
        # Strassen factors for 2x2x2
        # A indices: i in 0..1, j in j_offset..j_offset+1
        # B indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
        # C indices: i in 0..1, k in k_offset..k_offset+1
        
        # U (A)
        # M1: (a11+a22) -> U[i, col] = 1 if i=0,j=j_off or i=1,j=j_off+1
        # M2: (a21+a22) -> U[i, col] = 1 if i=1, j=j_off or j=j_off+1
        # M3: a11 -> U[0, j_off]
        # M4: a22 -> U[1, j_off+1]
        # M5: (a11+a12) -> U[0, j_off], U[0, j_off+1]
        # M6: (a21-a11) -> U[1, j_off]=1, U[0, j_off]=-1
        # M7: (a12-a22) -> U[0, j_off+1]=1, U[1, j_off+1]=-1
        
        # V (B)
        # M1: (b11+b22) -> V[j_off, k_off]=1, V[j_off+1, k_off+1]=1
        # M2: b11 -> V[j_off, k_off]=1
        # M3: (b12-b22) -> V[j_off, k_off+1]=1, V[j_off+1, k_off+1]=-1
        # M4: (b21-b11) -> V[j_off+1, k_off]=1, V[j_off, k_off]=-1
        # M5: b22 -> V[j_off+1, k_off+1]=1
        # M6: (b11+b12) -> V[j_off, k_off]=1, V[j_off, k_off+1]=1
        # M7: (b21+b22) -> V[j_off+1, k_off]=1, V[j_off+1, k_off+1]=1
        
        # W (C)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # Note: W index is k*n + i.
        # k is k_offset + 0 or 1. i is 0 or 1.
        
        # M1
        U[0 * m + j_offset][col_offset] += 1.0
        U[1 * m + j_offset + 1][col_offset] += 1.0
        V[j_offset * p + k_offset][col_offset] += 1.0
        V[(j_offset + 1) * p + k_offset + 1][col_offset] += 1.0
        W[(k_offset) * n + 0][col_offset] += 1.0
        W[(k_offset + 1) * n + 1][col_offset] += 1.0
        
        # M2
        U[1 * m + j_offset][col_offset + 1] += 1.0
        U[1 * m + j_offset + 1][col_offset + 1] += 1.0
        V[j_offset * p + k_offset][col_offset + 1] += 1.0
        W[(k_offset) * n + 1][col_offset + 1] += 1.0
        
        # M3
        U[0 * m + j_offset][col_offset + 2] += 1.0
        V[j_offset * p + k_offset + 1][col_offset + 2] += 1.0
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] += -1.0
        W[(k_offset + 1) * n + 0][col_offset + 2] += 1.0
        
        # M4
        U[1 * m + j_offset + 1][col_offset + 3] += 1.0
        V[(j_offset + 1) * p + k_offset][col_offset + 3] += 1.0
        V[j_offset * p + k_offset][col_offset + 3] += -1.0
        W[(k_offset) * n + 1][col_offset + 3] += 1.0
        W[(k_offset + 1) * n + 0][col_offset + 3] += 1.0 # Wait, c11 has +M4, c21 has +M4.
        
        # M5
        U[0 * m + j_offset][col_offset + 4] += 1.0
        U[0 * m + j_offset + 1][col_offset + 4] += 1.0
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] += 1.0
        W[(k_offset + 1) * n + 0][col_offset + 4] += -1.0
        W[(k_offset + 1) * n + 1][col_offset + 4] += 1.0
        
        # M6
        U[1 * m + j_offset][col_offset + 5] += 1.0
        U[0 * m + j_offset][col_offset + 5] += -1.0
        V[j_offset * p + k_offset][col_offset + 5] += 1.0
        V[j_offset * p + k_offset + 1][col_offset + 5] += 1.0
        W[(k_offset + 1) * n + 1][col_offset + 5] += 1.0
        
        # M7
        U[0 * m + j_offset + 1][col_offset + 6] += 1.0
        U[1 * m + j_offset + 1][col_offset + 6] += -1.0
        V[(j_offset + 1) * p + k_offset][col_offset + 6] += 1.0
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] += 1.0
        W[(k_offset) * n + 0][col_offset + 6] += 1.0

    # Helper to add a 2x2x1 block
    def add_2x2x1_block(j_offset, k_offset, col_offset):
        # A 2x2, B 2x1, C 2x1
        # c1 = a11*b11 + a12*b21
        # c2 = a21*b11 + a22*b21
        # 4 multiplications
        # M1: a11*b11
        U[0 * m + j_offset][col_offset] += 1.0
        V[j_offset * p + k_offset][col_offset] += 1.0
        W[k_offset * n + 0][col_offset] += 1.0
        
        # M2: a12*b21
        U[0 * m + j_offset + 1][col_offset + 1] += 1.0
        V[(j_offset + 1) * p + k_offset][col_offset + 1] += 1.0
        W[k_offset * n + 0][col_offset + 1] += 1.0
        
        # M3: a21*b11
        U[1 * m + j_offset][col_offset + 2] += 1.0
        V[j_offset * p + k_offset][col_offset + 2] += 1.0
        W[k_offset * n + 1][col_offset + 2] += 1.0
        
        # M4: a22*b21
        U[1 * m + j_offset + 1][col_offset + 3] += 1.0
        V[(j_offset + 1) * p + k_offset][col_offset + 3] += 1.0
        W[k_offset * n + 1][col_offset + 3] += 1.0

    # Block 1: j_offset=0
    # k_offset=0 (2x2x2)
    add_strassen_block(0, 0, 0)
    # k_offset=2 (2x2x2)
    add_strassen_block(0, 2, 7)
    # k_offset=4 (2x2x1)
    add_2x2x1_block(0, 4, 14)
    
    # Block 2: j_offset=2
    # k_offset=0 (2x2x2)
    add_strassen_block(2, 0, 18)
    # k_offset=2 (2x2x2)
    add_strassen_block(2, 2, 25)
    # k_offset=4 (2x2x1)
    add_2x2x1_block(2, 4, 32)
    
    return {"U": U, "V": V, "W": W}
