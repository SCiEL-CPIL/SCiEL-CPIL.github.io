import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will build the U, V, W matrices for the Strassen-based decomposition.
    # Rank 44.
    
    # Strassen decomposition for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global indices.
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset, j_offset, k_offset are the starting indices for the 2x2 block
        # A indices: (i_offset + di, j_offset + dj)
        # B indices: (j_offset + dj, k_offset + dk)
        # C indices: (i_offset + di, k_offset + dk)
        
        # M1
        # U: A00, A11
        U[i_offset * m + j_offset][col] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] = 1
        # V: B00, B11
        V[j_offset * p + k_offset][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # W: C00, C11
        W[k_offset * n + i_offset][col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # M2
        # U: A10, A11
        U[(i_offset + 1) * m + j_offset][col] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] = 1
        # V: B00
        V[j_offset * p + k_offset][col] = 1
        # W: C10
        W[k_offset * n + (i_offset + 1)][col] = 1
        col += 1
        
        # M3
        # U: A00
        U[i_offset * m + j_offset][col] = 1
        # V: B01, -B11
        V[j_offset * p + (k_offset + 1)][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = -1
        # W: C01
        W[(k_offset + 1) * n + i_offset][col] = 1
        col += 1
        
        # M4
        # U: A11
        U[(i_offset + 1) * m + (j_offset + 1)][col] = 1
        # V: B10, -B00
        V[(j_offset + 1) * p + k_offset][col] = 1
        V[j_offset * p + k_offset][col] = -1
        # W: C00, C10
        W[k_offset * n + i_offset][col] = 1
        W[k_offset * n + (i_offset + 1)][col] = 1
        col += 1
        
        # M5
        # U: A00, A01
        U[i_offset * m + j_offset][col] = 1
        U[i_offset * m + (j_offset + 1)][col] = 1
        # V: B11
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # W: C00, C01
        W[k_offset * n + i_offset][col] = -1
        W[(k_offset + 1) * n + i_offset][col] = 1
        col += 1
        
        # M6
        # U: A10, -A00
        U[(i_offset + 1) * m + j_offset][col] = 1
        U[i_offset * m + j_offset][col] = -1
        # V: B00, B01
        V[j_offset * p + k_offset][col] = 1
        V[j_offset * p + (k_offset + 1)][col] = 1
        # W: C11
        W[(k_offset + 1) * n + (i_offset + 1)][col] = 1
        col += 1
        
        # M7
        # U: A01, -A11
        U[i_offset * m + (j_offset + 1)][col] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col] = -1
        # V: B10, B11
        V[(j_offset + 1) * p + k_offset][col] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col] = 1
        # W: C00, C11
        W[k_offset * n + i_offset][col] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col] = 1
        col += 1
        
    def add_schoolbook_block_1x2x2(i_idx, j_start, k_start):
        nonlocal col
        # i is fixed, j in {j_start, j_start+1}, k in {k_start, k_start+1}
        # C_k = A_j B_jk
        # 4 terms
        for j in range(2):
            for k in range(2):
                U[i_idx * m + (j_start + j)][col] = 1
                V[(j_start + j) * p + (k_start + k)][col] = 1
                W[(k_start + k) * n + i_idx][col] = 1
                col += 1

    # 4 Strassen blocks for i in {0,1}
    # j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    # j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    # j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    # j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    
    # 4 Schoolbook blocks for i=2
    # j in {0,1}, k in {0,1}
    add_schoolbook_block_1x2x2(2, 0, 0)
    # j in {0,1}, k in {2,3}
    add_schoolbook_block_1x2x2(2, 0, 2)
    # j in {2,3}, k in {0,1}
    add_schoolbook_block_1x2x2(2, 2, 0)
    # j in {2,3}, k in {2,3}
    add_schoolbook_block_1x2x2(2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
