import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# Strassen decomposition for 2x2x2
# M1 = A00(B00+B11)
# M2 = (A00+A01)B11
# M3 = (A10+A11)(B00-B10)
# M4 = A11(B10+B00)
# M5 = (A00+A11)(B11-B10)
# M6 = (A01-A00)(B00+B10)
# M7 = (A10-A11)(B01+B11)

def construct():
    n, m, p = DIMENSIONS
    # We use rank 56 decomposition
    # 8 independent 2x2x2 Strassen decompositions
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    col = 0
    # A is 2x4. Split into A1 (2x2) and A2 (2x2).
    # B is 4x8. Split into B1 (2x8) and B2 (2x8).
    # B1 = [B11, B12] where B11, B12 are 2x4.
    # B2 = [B21, B22] where B21, B22 are 2x4.
    # C = A1 B1 + A2 B2
    # A1 B1 = A1 B11 + A1 B12
    # A2 B2 = A2 B21 + A2 B22
    
    # Each 2x2x4 is two 2x2x2.
    # So we have 4 * 2 = 8 2x2x2 blocks.
    
    # Indices for A: 0,1 (row), 2,3 (col)
    # A1: rows 0,1; cols 0,1
    # A2: rows 0,1; cols 2,3
    
    # Indices for B: 0,1 (row), 2,3 (row); 0..7 (col)
    # B1: rows 0,1; cols 0..7
    # B2: rows 2,3; cols 0..7
    # B11: rows 0,1; cols 0,1
    # B12: rows 0,1; cols 2,3
    # B13: rows 0,1; cols 4,5
    # B14: rows 0,1; cols 6,7
    # Similarly for B2.
    
    # Wait, B is 4x8.
    # B1 is top 2 rows, all 8 cols.
    # B2 is bottom 2 rows, all 8 cols.
    
    # We need to compute A1 * B1 (2x2 * 2x8 -> 2x8)
    # and A2 * B2 (2x2 * 2x8 -> 2x8)
    
    # A1 * B1 can be done by splitting B1 into four 2x2 blocks.
    # B1 = [B1_0, B1_1, B1_2, B1_3]
    # A1 * B1 = [A1 B1_0, A1 B1_1, A1 B1_2, A1 B1_3]
    
    # Each A1 B1_k is 2x2x2.
    
    # Let's define the 8 blocks
    # Block 0: A1, B1_0 (cols 0,1 of B1)
    # Block 1: A1, B1_1 (cols 2,3 of B1)
    # Block 2: A1, B1_2 (cols 4,5 of B1)
    # Block 3: A1, B1_3 (cols 6,7 of B1)
    # Block 4: A2, B2_0 (cols 0,1 of B2)
    # Block 5: A2, B2_1 (cols 2,3 of B2)
    # Block 6: A2, B2_2 (cols 4,5 of B2)
    # Block 7: A2, B2_3 (cols 6,7 of B2)
    
    # Strassen terms for 2x2x2:
    # U vectors (for A):
    # M1: A00
    # M2: A00+A01
    # M3: A10+A11
    # M4: A11
    # M5: A00+A11
    # M6: A01-A00
    # M7: A10-A11
    
    # V vectors (for B):
    # M1: B00+B11
    # M2: B11
    # M3: B00-B10
    # M4: B10+B00
    # M5: B11-B10
    # M6: B00+B10
    # M7: B01+B11
    
    # W vectors (for C):
    # C00 = M1 - M2 + M5 - M6
    # C01 = M2 + M7
    # C10 = M3 + M4
    # C11 = M1 - M5 + M4 - M3
    
    # We will fill U, V, W for each of the 8 blocks.
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        nonlocal col
        # a_row_offset, a_col_offset: top-left of 2x2 A block in U (size 8)
        # b_row_offset, b_col_offset: top-left of 2x2 B block in V (size 32)
        # c_row_offset, c_col_offset: top-left of 2x2 C block in W (size 16)
        
        # U indices:
        # A00: a_row_offset, a_col_offset
        # A01: a_row_offset, a_col_offset+1
        # A10: a_row_offset+1, a_col_offset
        # A11: a_row_offset+1, a_col_offset+1
        
        # V indices:
        # B00: b_row_offset, b_col_offset
        # B01: b_row_offset, b_col_offset+1
        # B10: b_row_offset+1, b_col_offset
        # B11: b_row_offset+1, b_col_offset+1
        
        # W indices:
        # C00: c_row_offset, c_col_offset
        # C01: c_row_offset, c_col_offset+1
        # C10: c_row_offset+1, c_col_offset
        # C11: c_row_offset+1, c_col_offset+1
        
        # M1
        U[a_row_offset][start_col] = 1
        U[a_row_offset+1][start_col] = 0 # A10 is 0? No, A is 2x2.
        # Wait, U is for A.
        # M1 uses A00.
        U[a_row_offset][start_col] = 1
        U[a_row_offset+1][start_col] = 0 # A10 coefficient is 0
        # Actually, let's just map the linear forms.
        
        # M1: A00
        U[a_row_offset][start_col] = 1
        U[a_row_offset+1][start_col] = 0
        U[a_row_offset+2][start_col] = 0 # A is 2x4, so rows 0,1.
        # Wait, U has 8 rows.
        # A1 is rows 0,1. A2 is rows 0,1 but cols 2,3.
        # So for A1, rows are 0,1. Cols are 0,1.
        # U index: row = a_row, col = a_col.
        
        # Let's use explicit coordinates.
        # A1: rows 0,1; cols 0,1.
        # A2: rows 0,1; cols 2,3.
        
        # M1: A00
        U[a_row_offset][start_col] = 1
        U[a_row_offset+1][start_col] = 0
        # V: B00+B11
        V[b_row_offset][start_col] = 1
        V[b_row_offset+1][start_col] = 1
        # W: C00
        W[c_row_offset][start_col] = 1
        W[c_row_offset+1][start_col] = 0 # C10 is 0 for M1
        
        col += 1
        
        # M2: A00+A01
        U[a_row_offset][start_col] = 1
        U[a_row_offset+1][start_col] = 0
        # V: B11
        V[b_row_offset][start_col] = 0
        V[b_row_offset+1][start_col] = 1
        # W: C01
        W[c_row_offset][start_col] = 0
        W[c_row_offset+1][start_col] = 1
        
        col += 1
        
        # M3: A10+A11
        U[a_row_offset][start_col] = 0
        U[a_row_offset+1][start_col] = 1
        # V: B00-B10
        V[b_row_offset][start_col] = 1
        V[b_row_offset+1][start_col] = -1
        # W: C10
        W[c_row_offset][start_col] = 0
        W[c_row_offset+1][start_col] = 1
        
        col += 1
        
        # M4: A11
        U[a_row_offset][start_col] = 0
        U[a_row_offset+1][start_col] = 1
        # V: B10+B00
        V[b_row_offset][start_col] = 1
        V[b_row_offset+1][start_col] = 1
        # W: C11
        W[c_row_offset][start_col] = 0
        W[c_row_offset+1][start_col] = 0 # C11 is row 1 col 1 of C block
        # Wait, W index for C11 is c_row_offset+1, c_col_offset+1
        # My W update above was for C10.
        # Let's fix W updates.
        
        # I will rewrite the block filling more carefully.
        
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 0: A1 (0,0), B1_0 (0,0), C_0 (0,0)
    # A1: U[0:2, 0:2]
    # B1_0: V[0:2, 0:2]
    # C_0: W[0:2, 0:2]
    
    # I will create a helper to add a 2x2x2 Strassen decomposition
    # given the top-left indices in U, V, W.
    
    def add_2x2x2(u_r, u_c, v_r, v_c, w_r, w_c, start):
        nonlocal col
        # M1: A00(B00+B11) -> C00
        U[u_r][start] = 1; U[u_r+1][start] = 0
        V[v_r][start] = 1; V[v_r+1][start] = 1
        W[w_r][start] = 1; W[w_r+1][start] = 0
        col += 1
        
        # M2: (A00+A01)B11 -> C01
        U[u_r][start] = 1; U[u_r+1][start] = 0
        V[v_r][start] = 0; V[v_r+1][start] = 1
        W[w_r][start] = 0; W[w_r+1][start] = 1
        col += 1
        
        # M3: (A10+A11)(B00-B10) -> C10
        U[u_r][start] = 0; U[u_r+1][start] = 1
        V[v_r][start] = 1; V[v_r+1][start] = -1
        W[w_r][start] = 0; W[w_r+1][start] = 1
        col += 1
        
        # M4: A11(B10+B00) -> C11
        U[u_r][start] = 0; U[u_r+1][start] = 1
        V[v_r][start] = 1; V[v_r+1][start] = 1
        W[w_r][start] = 0; W[w_r+1][start] = 0 # C11 is at w_r+1, w_c+1
        # Wait, C11 is row 1, col 1 of the 2x2 C block.
        # So W[w_r+1][start] = 0 is wrong if I want to set C11.
        # I need to set W[w_r+1][start] = 1? No, W is p*n x rank.
        # W index is k*n+i.
        # C00: k=0, i=0 -> W[0]
        # C01: k=0, i=1 -> W[1]
        # C10: k=1, i=0 -> W[2]
        # C11: k=1, i=1 -> W[3]
        # So for C11, I should set W[w_r+1][start] = 1?
        # w_r is the row index in W.
        # If w_r is 0, C11 is at W[3].
        # So W[w_r+1] is W[1]. That's C01.
        # I need W[w_r+2] for C10 and W[w_r+3] for C11?
        # No, W is flattened.
        # Let's assume W is stored as [C00, C01, C10, C11] for a 2x2 block.
        # So W[w_r] = C00, W[w_r+1] = C01, W[w_r+2] = C10, W[w_r+3] = C11.
        
        # Let's re-do W updates.
        # M1 -> C00: W[w_r] += 1
        # M2 -> C01: W[w_r+1] += 1
        # M3 -> C10: W[w_r+2] += 1
        # M4 -> C11: W[w_r+3] += 1
        # M5 -> C00: W[w_r] += 1, C11: W[w_r+3] -= 1
        # M6 -> C00: W[w_r] -= 1, C10: W[w_r+2] -= 1
        # M7 -> C01: W[w_r+1] += 1, C11: W[w_r+3] -= 1
        
        # Let's implement this correctly.
        
        # I will restart the function body to be safe.
        
    # Re-initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_2x2x2(u_r, u_c, v_r, v_c, w_r, w_c, start):
        nonlocal col
        # M1: A00(B00+B11) -> C00
        U[u_r][start] = 1
        V[v_r][start] = 1
        V[v_r+1][start] = 1
        W[w_r][start] = 1
        col += 1
        
        # M2: (A00+A01)B11 -> C01
        U[u_r][start] = 1
        U[u_r+1][start] = 0 # A01 is at u_r, u_c+1
        # Wait, U is flattened.
        # A00: u_r, u_c
        # A01: u_r, u_c+1
        # A10: u_r+1, u_c
        # A11: u_r+1, u_c+1
        
        # M2 uses A00+A01.
        # So U[u_r][start] += 1 (for A00)
        # U[u_r+1][start] += 0 (for A10)
        # But A01 is at U[u_r+1]? No, A is 2x2.
        # U has 8 rows.
        # A1: rows 0,1.
        # A00: U[0], A01: U[1]? No.
        # A is 2x4.
        # A1 is 2x2.
        # A1[0,0] is U[0]. A1[0,1] is U[1].
        # A1[1,0] is U[2]. A1[1,1] is U[3].
        # So A00=U[0], A01=U[1], A10=U[2], A11=U[3].
        
        # Let's use this mapping.
        # u_r is the row index in U.
        # A00: U[u_r]
        # A01: U[u_r+1]
        # A10: U[u_r+2]
        # A11: U[u_r+3]
        
        # Similarly for V.
        # v_r is row index in V.
        # B00: V[v_r]
        # B01: V[v_r+1]
        # B10: V[v_r+2]
        # B11: V[v_r+3]
        
        # Similarly for W.
        # w_r is row index in W.
        # C00: W[w_r]
        # C01: W[w_r+1]
        # C10: W[w_r+2]
        # C11: W[w_r+3]
        
        # M1: A00(B00+B11) -> C00
        U[u_r][start] = 1
        V[v_r][start] = 1
        V[v_r+3][start] = 1
        W[w_r][start] = 1
        col += 1
        
        # M2: (A00+A01)B11 -> C01
        U[u_r][start] = 1
        U[u_r+1][start] = 1
        V[v_r+3][start] = 1
        W[w_r+1][start] = 1
        col += 1
        
        # M3: (A10+A11)(B00-B10) -> C10
        U[u_r+2][start] = 1
        U[u_r+3][start] = 1
        V[v_r][start] = 1
        V[v_r+2][start] = -1
        W[w_r+2][start] = 1
        col += 1
        
        # M4: A11(B10+B00) -> C11
        U[u_r+3][start] = 1
        V[v_r+2][start] = 1
        V[v_r][start] = 1
        W[w_r+3][start] = 1
        col += 1
        
        # M5: (A00+A11)(B11-B10) -> C00, C11
        U[u_r][start] = 1
        U[u_r+3][start] = 1
        V[v_r+3][start] = 1
        V[v_r+2][start] = -1
        W[w_r][start] = 1
        W[w_r+3][start] = -1
        col += 1
        
        # M6: (A01-A00)(B00+B10) -> C00, C10
        U[u_r+1][start] = 1
        U[u_r][start] = -1
        V[v_r][start] = 1
        V[v_r+2][start] = 1
        W[w_r][start] = -1
        W[w_r+2][start] = -1
        col += 1
        
        # M7: (A10-A11)(B01+B11) -> C01, C11
        U[u_r+2][start] = 1
        U[u_r+3][start] = -1
        V[v_r+1][start] = 1
        V[v_r+3][start] = 1
        W[w_r+1][start] = 1
        W[w_r+3][start] = -1
        col += 1

    # 8 blocks
    # A1: u_r=0
    # A2: u_r=2
    # B1_0: v_r=0
    # B1_1: v_r=2
    # B1_2: v_r=4
    # B1_3: v_r=6
    # B2_0: v_r=8
    # B2_1: v_r=10
    # B2_2: v_r=12
    # B2_3: v_r=14
    
    # C_0: w_r=0
    # C_1: w_r=2
    # C_2: w_r=4
    # C_3: w_r=6
    # C_4: w_r=8
    # C_5: w_r=10
    # C_6: w_r=12
    # C_7: w_r=14
    
    # Block 0: A1, B1_0, C_0
    add_2x2x2(0, 0, 0, 0, 0, 0, col)
    # Block 1: A1, B1_1, C_1
    add_2x2x2(0, 0, 2, 0, 2, 0, col)
    # Block 2: A1, B1_2, C_2
    add_2x2x2(0, 0, 4, 0, 4, 0, col)
    # Block 3: A1, B1_3, C_3
    add_2x2x2(0, 0, 6, 0, 6, 0, col)
    
    # Block 4: A2, B2_0, C_4
    add_2x2x2(2, 0, 8, 0, 8, 0, col)
    # Block 5: A2, B2_1, C_5
    add_2x2x2(2, 0, 10, 0, 10, 0, col)
    # Block 6: A2, B2_2, C_6
    add_2x2x2(2, 0, 12, 0, 12, 0, col)
    # Block 7: A2, B2_3, C_7
    add_2x2x2(2, 0, 14, 0, 14, 0, col)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
