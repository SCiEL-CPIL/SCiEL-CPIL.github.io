import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition using Strassen for 2x2 blocks
    # 5 blocks of 2 columns, 1 block of 1 column
    num_strassen_blocks = 5
    num_schoolbook_blocks = 1
    rank = num_strassen_blocks * 7 + num_schoolbook_blocks * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    rank_offset = 0
    
    # Strassen blocks
    for block_idx in range(num_strassen_blocks):
        # B block columns: col_offset, col_offset + 1
        # A is full 2x2
        
        # Strassen terms:
        # P1 = A11 * (B11 + B21)  -- Note: B indices in block are local (0,1)
        # Let's map global B indices
        # B row 0, col c0 -> V[0*p + c0]
        # B row 1, col c1 -> V[1*p + c1]
        
        c0 = col_offset
        c1 = col_offset + 1
        
        # Term 1: A11 * (B1,c0 + B1,c1)
        # U: A11 is at index 0*2 + 0 = 0
        # V: B1,c0 at 0*11 + c0, B1,c1 at 0*11 + c1
        # W: C1,c0 at c0*2 + 0, C1,c1 at c1*2 + 0
        # Wait, W index is k*n + i. k is col, i is row.
        # C[i,k].
        # Term contributes to C[0, c0] and C[0, c1]?
        # P1 = A11 * (B1,c0 + B1,c1)
        # This product is part of C[0, c0] = A11*B1,c0 + A12*B2,c0
        # And C[0, c1] = A11*B1,c1 + A12*B2,c1
        # So P1 contributes A11*B1,c0 to C[0,c0] and A11*B1,c1 to C[0,c1].
        # So W should have 1 at (c0, 0) and (c1, 0).
        
        U[0][rank_offset] = 1
        V[0*p + c0][rank_offset] = 1
        V[0*p + c1][rank_offset] = 1
        W[c0*n + 0][rank_offset] = 1
        W[c1*n + 0][rank_offset] = 1
        rank_offset += 1
        
        # Term 2: (A21 + A22) * B1,c0
        # Contributes to C[1, c0]
        U[1*m + 1][rank_offset] = 1 # A21
        U[1*m + 2][rank_offset] = 1 # A22
        V[0*p + c0][rank_offset] = 1
        W[c0*n + 1][rank_offset] = 1
        rank_offset += 1
        
        # Term 3: A11 * (B1,c1 - B1,c0)
        # Contributes to C[0, c1] and C[0, c0]
        U[0][rank_offset] = 1
        V[0*p + c1][rank_offset] = 1
        V[0*p + c0][rank_offset] = -1
        W[c1*n + 0][rank_offset] = 1
        W[c0*n + 0][rank_offset] = -1
        rank_offset += 1
        
        # Term 4: A22 * (B2,c0 - B2,c1)
        # Contributes to C[1, c0] and C[1, c1]
        U[1*m + 2][rank_offset] = 1
        V[1*p + c0][rank_offset] = 1
        V[1*p + c1][rank_offset] = -1
        W[c0*n + 1][rank_offset] = 1
        W[c1*n + 1][rank_offset] = -1
        rank_offset += 1
        
        # Term 5: (A11 + A12) * B2,c1
        # Contributes to C[0, c1]
        U[0][rank_offset] = 1
        U[0*m + 1][rank_offset] = 1
        V[1*p + c1][rank_offset] = 1
        W[c1*n + 0][rank_offset] = 1
        rank_offset += 1
        
        # Term 6: (A21 - A11) * (B1,c0 + B1,c1)
        # Contributes to C[1, c0] and C[1, c1]
        U[1*m + 1][rank_offset] = 1
        U[0][rank_offset] = -1
        V[0*p + c0][rank_offset] = 1
        V[0*p + c1][rank_offset] = 1
        W[c0*n + 1][rank_offset] = 1
        W[c1*n + 1][rank_offset] = 1
        rank_offset += 1
        
        # Term 7: (A12 - A22) * (B2,c0 + B2,c1)
        # Contributes to C[0, c0] and C[0, c1]
        U[0*m + 1][rank_offset] = 1
        U[1*m + 2][rank_offset] = -1
        V[1*p + c0][rank_offset] = 1
        V[1*p + c1][rank_offset] = 1
        W[c0*n + 0][rank_offset] = 1
        W[c1*n + 0][rank_offset] = 1
        rank_offset += 1
        
        col_offset += 2
        
    # Schoolbook block for last column
    c0 = col_offset
    # C[0, c0] = A11*B1,c0 + A12*B2,c0
    # C[1, c0] = A21*B1,c0 + A22*B2,c0
    
    # Term 1: A11 * B1,c0
    U[0][rank_offset] = 1
    V[0*p + c0][rank_offset] = 1
    W[c0*n + 0][rank_offset] = 1
    rank_offset += 1
    
    # Term 2: A12 * B2,c0
    U[0*m + 1][rank_offset] = 1
    V[1*p + c0][rank_offset] = 1
    W[c0*n + 0][rank_offset] = 1
    rank_offset += 1
    
    # Term 3: A21 * B1,c0
    U[1*m + 1][rank_offset] = 1
    V[0*p + c0][rank_offset] = 1
    W[c0*n + 1][rank_offset] = 1
    rank_offset += 1
    
    # Term 4: A22 * B2,c0
    U[1*m + 2][rank_offset] = 1
    V[1*p + c0][rank_offset] = 1
    W[c0*n + 1][rank_offset] = 1
    rank_offset += 1
    
    return {"U": U, "V": V, "W": W}
