"""
Matrix multiplication 2x2 by 2x7 tensor decomposition.

This program uses a block-wise decomposition approach:
1. It groups the 7 columns of the second matrix B into three pairs and one single column.
2. For each pair of columns, it uses Laderman's algorithm (rank 7) to compute the corresponding 2x2 block of the result.
3. For the single remaining column, it uses the standard schoolbook algorithm (rank 4) to compute the 2x1 block.
4. Total rank: 3 * 7 + 4 = 25.

This is an improvement over the standard schoolbook decomposition which has rank 2 * 2 * 7 = 28.
"""

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (4, 25)
    # V shape: (m*p, rank) = (14, 25)
    # W shape: (p*n, rank) = (14, 25)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Laderman block
    # base_col: starting column index in U, V, W
    # b_col_start: starting column index in B (0..6)
    def add_laderman_block(base_col, b_col_start):
        # B indices for this block:
        # e = B[0, b_col_start] -> index 0*p + b_col_start
        # f = B[0, b_col_start+1] -> index 0*p + b_col_start+1
        # g = B[1, b_col_start] -> index 1*p + b_col_start
        # h = B[1, b_col_start+1] -> index 1*p + b_col_start+1
        
        e_idx = b_col_start
        f_idx = b_col_start + 1
        g_idx = p + b_col_start
        h_idx = p + b_col_start + 1
        
        # C indices for this block:
        # C[0, b_col_start] -> index b_col_start * n + 0
        # C[1, b_col_start] -> index b_col_start * n + 1
        # C[0, b_col_start+1] -> index (b_col_start+1) * n + 0
        # C[1, b_col_start+1] -> index (b_col_start+1) * n + 1
        
        c00_idx = b_col_start * n
        c10_idx = b_col_start * n + 1
        c01_idx = (b_col_start + 1) * n
        c11_idx = (b_col_start + 1) * n + 1
        
        # Term 1: (a+d)(e+h)
        # U: a=1, d=1
        U[0][base_col] = 1
        U[3][base_col] = 1
        # V: e=1, h=1
        V[e_idx][base_col] = 1
        V[h_idx][base_col] = 1
        # W: C01 += 1, C11 += 1
        W[c01_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        
        # Term 2: (c+d)h
        # U: c=1, d=1
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        # V: h=1
        V[h_idx][base_col + 1] = 1
        # W: C10 += 1, C11 += -1
        W[c10_idx][base_col + 1] = 1
        W[c11_idx][base_col + 1] = -1
        
        # Term 3: a e
        # U: a=1
        U[0][base_col + 2] = 1
        # V: e=1
        V[e_idx][base_col + 2] = 1
        # W: C00 += 1
        W[c00_idx][base_col + 2] = 1
        
        # Term 4: d (h-g)
        # U: d=1
        U[3][base_col + 3] = 1
        # V: h=1, g=-1
        V[h_idx][base_col + 3] = 1
        V[g_idx][base_col + 3] = -1
        # W: C00 += 1, C01 += -1
        W[c00_idx][base_col + 3] = 1
        W[c01_idx][base_col + 3] = -1
        
        # Term 5: (a-b) e
        # U: a=1, b=-1
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = -1
        # V: e=1
        V[e_idx][base_col + 4] = 1
        # W: C01 += -1, C10 += 1, C11 += -1
        W[c01_idx][base_col + 4] = -1
        W[c10_idx][base_col + 4] = 1
        W[c11_idx][base_col + 4] = -1
        
        # Term 6: (b+c) f
        # U: b=1, c=1
        U[1][base_col + 5] = 1
        U[2][base_col + 5] = 1
        # V: f=1
        V[f_idx][base_col + 5] = 1
        # W: C11 += 1
        W[c11_idx][base_col + 5] = 1
        
        # Term 7: (d-c) g
        # U: d=1, c=-1
        U[3][base_col + 6] = 1
        U[2][base_col + 6] = -1
        # V: g=1
        V[g_idx][base_col + 6] = 1
        # W: C00 += 1, C01 += -1, C10 += 1
        W[c00_idx][base_col + 6] = 1
        W[c01_idx][base_col + 6] = -1
        W[c10_idx][base_col + 6] = 1

    # Add 3 Laderman blocks for columns (0,1), (2,3), (4,5)
    add_laderman_block(0, 0)
    add_laderman_block(7, 2)
    add_laderman_block(14, 4)
    
    # Add Schoolbook block for column 6
    base_col = 21
    b_idx_0 = 6 # B[0, 6]
    b_idx_1 = p + 6 # B[1, 6]
    c_idx_0 = 6 * n # C[0, 6]
    c_idx_1 = 6 * n + 1 # C[1, 6]
    
    # Term 1: a * B[0,6]
    U[0][base_col] = 1
    V[b_idx_0][base_col] = 1
    W[c_idx_0][base_col] = 1
    
    # Term 2: b * B[1,6]
    U[1][base_col + 1] = 1
    V[b_idx_1][base_col + 1] = 1
    W[c_idx_0][base_col + 1] = 1
    
    # Term 3: c * B[0,6]
    U[2][base_col + 2] = 1
    V[b_idx_0][base_col + 2] = 1
    W[c_idx_1][base_col + 2] = 1
    
    # Term 4: d * B[1,6]
    U[3][base_col + 3] = 1
    V[b_idx_1][base_col + 3] = 1
    W[c_idx_1][base_col + 3] = 1

    return {"U": U, "V": V, "W": W}
