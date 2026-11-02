import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication with Strassen rank-7.
    U: shape (4, 7) - input A flattened (row-major: A00, A01, A10, A11)
    V: shape (4, 7) - input B flattened (row-major: B00, B01, B10, B11)
    W: shape (4, 7) - output C flattened (row-major: C00, C01, C10, C11)
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U[i, t] where i = row*2+col of A
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # t=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # t=1: M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    # C10 += M2, C11 -= M2
    W[2][1] = 1  # C10
    W[3][1] = -1  # C11
    
    # t=2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    # C01 += M3, C11 += M3
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # t=3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    # C00 += M4, C10 += M4
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # t=4: M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01
    
    # t=5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # C11 += M6
    W[3][5] = 1  # C11
    
    # t=6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    # C00 += M7
    W[0][6] = 1  # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=6
    deadline = time.time() + 110
    
    # Strategy: Split the 3x4x6 multiplication into blocks.
    # A is 3x4, B is 4x6.
    # Split A into A_top (2x4) and A_bottom (1x4).
    # Split B into B_left (4x3) and B_right (4x3).
    # 
    # C = A*B where C is 3x6.
    # C_top_left = A_top * B_left (2x3)
    # C_top_right = A_top * B_right (2x3)
    # C_bottom_left = A_bottom * B_left (1x3)
    # C_bottom_right = A_bottom * B_right (1x3)
    #
    # For 2x3 multiplication: rank = ?
    # 2x3 = 2x2x3 can be done as two 2x2x2 Strassen = rank 14.
    # Or more precisely, 2x3x1... no, it's 2x2x3.
    # Split B_left (4x3) into two 4x2 blocks? No, that doesn't help directly.
    #
    # Actually, let's split differently.
    # Split the middle dimension (m=4) into two parts of size 2.
    # A is 3x4 = [A_left (3x2) | A_right (3x2)]
    # B is 4x6 = [B_top (2x6); B_bottom (2x6)]
    #
    # C = A_left * B_top + A_right * B_bottom
    # C is 3x6.
    # A_left (3x2) * B_top (2x6) -> 3x6 product, rank = 3*2*6 = 36 schoolbook.
    # A_right (3x2) * B_bottom (2x6) -> 3x6 product, rank = 3*2*6 = 36 schoolbook.
    # Total: 72. No improvement.
    #
    # Let's use Strassen on the 2x2 blocks of the middle dimension.
    # Split A (3x4) into rows: A_top (2x4), A_bottom (1x4).
    # Split B (4x6) into cols: B_left (4x3), B_right (4x3).
    #
    # C_top_left (2x3) = A_top (2x4) * B_left (4x3)
    #   Split A_top into A_top_left (2x2), A_top_right (2x2)
    #   Split B_left into B_left_top (2x3), B_left_bottom (2x3)
    #   C_top_left = A_top_left * B_left_top + A_top_right * B_left_bottom
    #   A_top_left (2x2) * B_left_top (2x3): This is 2x2x3.
    #     Split B_left_top into two 2x2 blocks: B_lt0 (2x2), B_lt1 (2x1)
    #     A_top_left * B_lt0 is 2x2x2, rank 7.
    #     A_top_left * B_lt1 is 2x2x1, rank 4 (schoolbook: 2*2*1=4).
    #     Total for this: 11.
    #   Similarly A_top_right * B_left_bottom: 11.
    #   Total for C_top_left: 22.
    #
    # C_top_right (2x3) = A_top (2x4) * B_right (4x3)
    #   Similarly: 22.
    #
    # C_bottom_left (1x3) = A_bottom (1x4) * B_left (4x3)
    #   A_bottom is 1x4. Split into A_bl (1x2), A_br (1x2).
    #   B_left split into B_lt (2x3), B_lb (2x3).
    #   C_bottom_left = A_bl * B_lt + A_br * B_lb
    #   A_bl (1x2) * B_lt (2x3): 1x2x3, rank 6 schoolbook.
    #   A_br (1x2) * B_lb (2x3): 1x2x3, rank 6.
    #   Total: 12.
    #
    # C_bottom_right (1x3) = A_bottom (1x4) * B_right (4x3)
    #   Similarly: 12.
    #
    # Total rank: 22 + 22 + 12 + 12 = 68. Slight improvement over 72.
    #
    # Can we do better? For 2x2x3, is there a rank < 11?
    # 2x2x3: The tensor is M_{2,2,3}. We can think of it as M_{2,2,2} + M_{2,2,1}.
    # Rank(M_{2,2,2}) = 7, Rank(M_{2,2,1}) = 4. Total 11.
    # Is there a better way? Probably not significantly.
    #
    # Let's try a different split. What if we use Strassen for the entire 2x4x3?
    # 2x4x3: Split into 2x2x3 + 2x2x3 = 11 + 11 = 22. Same.
    #
    # Let's just implement the rank-68 decomposition.
    # Actually, let me reconsider. The schoolbook for 3x4x6 is 72.
    # Getting 68 is an improvement but small.
    #
    # Let me try to use Strassen more aggressively.
    # What if I view 3x4x6 as involving a 3x3x3 core?
    # 3x3x3 has rank 23 (Laderman's algorithm).
    # 3x4x6 = 3x3x3 + 3x1x6 + 3x3x3? No, that's not right.
    #
    # Let's stick with the block approach.
    # A (3x4) = [A1 (2x4); A2 (1x4)]
    # B (4x6) = [B1 (4x3) B2 (4x3)]
    #
    # C11 = A1 * B1 (2x3)
    # C12 = A1 * B2 (2x3)
    # C21 = A2 * B1 (1x3)
    # C22 = A2 * B2 (1x3)
    #
    # For 2x3:
    #   A1 (2x4) * B1 (4x3)
    #   Split A1 = [A1a (2x2) A1b (2x2)]
    #   Split B1 = [B1a (2x3); B1b (2x3)]
    #   C11 = A1a * B1a + A1b * B1b
    #   A1a (2x2) * B1a (2x3): rank 11
    #   A1b (2x2) * B1b (2x3): rank 11
    #   Total: 22
    #
    # For 1x3:
    #   A2 (1x4) * B1 (4x3)
    #   Split A2 = [A2a (1x2) A2b (1x2)]
    #   Split B1 = [B1a (2x3); B1b (2x3)]
    #   C21 = A2a * B1a + A2b * B1b
    #   A2a (1x2) * B1a (2x3): rank 6
    #   A2b (1x2) * B1b (2x3): rank 6
    #   Total: 12
    #
    # Grand total: 22 + 22 + 12 + 12 = 68.
    
    # Helper to add a block multiplication to the global U, V, W
    def add_block_mult(U_global, V_global, W_global, 
                       A_row_offset, A_col_offset, A_shape,
                       B_row_offset, B_col_offset, B_shape,
                       C_row_offset, C_col_offset, C_shape,
                       block_U, block_V, block_W,
                       next_col):
        """
        Add a block multiplication to the global decomposition.
        A_shape = (a_rows, a_cols), B_shape = (b_rows, b_cols), C_shape = (c_rows, c_cols)
        block_U, block_V, block_W are for the block multiplication.
        """
        a_rows, a_cols = A_shape
        b_rows, b_cols = B_shape
        c_rows, c_cols = C_shape
        rank = block_U.shape[1]
        
        for t in range(rank):
            for i in range(a_rows):
                for j in range(a_cols):
                    val = block_U[i*a_cols + j, t]
                    if val != 0:
                        U_global[A_row_offset + i, A_col_offset + j, next_col + t] += val
            for i in range(b_rows):
                for j in range(b_cols):
                    val = block_V[i*b_cols + j, t]
                    if val != 0:
                        V_global[B_row_offset + i, B_col_offset + j, next_col + t] += val
            for i in range(c_rows):
                for j in range(c_cols):
                    val = block_W[i*c_cols + j, t]
                    if val != 0:
                        W_global[C_row_offset + i, C_col_offset + j, next_col + t] += val
        return next_col + rank
    
    # Create global arrays
    total_rank = 68
    U_global = np.zeros((n*m, total_rank))
    V_global = np.zeros((m*p, total_rank))
    W_global = np.zeros((p*n, total_rank))
    
    # Get Strassen 2x2x2 decomposition
    sU, sV, sW = _strassen_2x2()
    sU = np.array(sU)
    sV = np.array(sV)
    sW = np.array(sW)
    
    # Helper for 2x2x3 decomposition (rank 11)
    def strassen_2x2x3():
        """
        Compute A (2x2) * B (2x3) using Strassen for the first 2 columns + schoolbook for the last.
        Returns (U, V, W) with rank 11.
        U: (4, 11), V: (6, 11), W: (6, 11)
        """
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        # First 7 columns: Strassen for 2x2x2 (columns 0,1 of B)
        for t in range(7):
            U[:, t] = sU[:, t]
            V[:4, t] = sV[:, t]  # B00, B01, B10, B11
            W[:4, t] = sW[:, t]   # C00, C01, C10, C11
        # Last 4 columns: schoolbook for 2x2x1 (column 2 of B)
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        # t=7: A00*B02 -> C02
        U[0, 7] = 1
        V[4, 7] = 1  # B02 (index 2*2+0=4)
        W[4, 7] = 1  # C02 (index 2*2+0=4)
        # t=8: A01*B12 -> C02
        U[1, 8] = 1
        V[5, 8] = 1  # B12 (index 2*2+1=5)
        W[4, 8] = 1
        # t=9: A10*B02 -> C12
        U[2, 9] = 1
        V[4, 9] = 1
        W[5, 9] = 1  # C12 (index 2*2+1=5)
        # t=10: A11*B12 -> C12
        U[3, 10] = 1
        V[5, 10] = 1
        W[5, 10] = 1
        return U, V, W
    
    # Helper for 1x2x3 decomposition (rank 6)
    def schoolbook_1x2x3():
        """
        Compute A (1x2) * B (2x3). Schoolbook rank 6.
        Returns (U, V, W) with rank 6.
        U: (2, 6), V: (6, 6), W: (3, 6)
        """
        U = np.zeros((2, 6))
        V = np.zeros((6, 6))
        W = np.zeros((3, 6))
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        # C2 = A0*B02 + A1*B12
        # t=0: A0*B00 -> C0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        # t=1: A1*B10 -> C0
        U[1, 1] = 1
        V[3, 1] = 1
        W[0, 1] = 1
        # t=2: A0*B01 -> C1
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        # t=3: A1*B11 -> C1
        U[1, 3] = 1
        V[4, 3] = 1
        W[1, 3] = 1
        # t=4: A0*B02 -> C2
        U[0, 4] = 1
        V[2, 4] = 1
        W[2, 4] = 1
        # t=5: A1*B12 -> C2
        U[1, 5] = 1
        V[5, 5] = 1
        W[2, 5] = 1
        return U, V, W
    
    next_col = 0
    
    # C11 = A1 * B1 (2x3) where A1 is rows 0,1 of A (2x4), B1 is cols 0,1,2 of B (4x3)
    # A1 = [A1a (2x2) A1b (2x2)]
    # B1 = [B1a (2x3); B1b (2x3)]
    # C11 = A1a * B1a + A1b * B1b
    
    # A1a * B1a: A1a is A[0:2, 0:2], B1a is B[0:2, 0:3]
    s2x2x3_U, s2x2x3_V, s2x2x3_W = strassen_2x2x3()
    next_col = add_block_mult(U_global, V_global, W_global,
                               0, 0, (2, 2),  # A1a: rows 0-1, cols 0-1
                               0, 0, (2, 3),  # B1a: rows 0-1, cols 0-2
                               0, 0, (2, 3),  # C11a: rows 0-1, cols 0-2
                               s2x2x3_U, s2x2x3_V, s2x2x3_W, next_col)
    
    # A1b * B1b: A1b is A[0:2, 2:4], B1b is B[2:4, 0:3]
    next_col = add_block_mult(U_global, V_global, W_global,
                               0, 2, (2, 2),  # A1b: rows 0-1, cols 2-3
                               2, 0, (2, 3),  # B1b: rows 2-3, cols 0-2
                               0, 0, (2, 3),  # C11b: rows 0-1, cols 0-2
                               s2x2x3_U, s2x2x3_V, s2x2x3_W, next_col)
    
    # C12 = A1 * B2 (2x3) where B2 is cols 3,4,5 of B
    # A1a * B2a: A1a is A[0:2, 0:2], B2a is B[0:2, 3:6]
    next_col = add_block_mult(U_global, V_global, W_global,
                               0, 0, (2, 2),  # A1a
                               0, 3, (2, 3),  # B2a: rows 0-1, cols 3-5
                               0, 3, (2, 3),  # C12a: rows 0-1, cols 3-5
                               s2x2x3_U, s2x2x3_V, s2x2x3_W, next_col)
    
    # A1b * B2b: A1b is A[0:2, 2:4], B2b is B[2:4, 3:6]
    next_col = add_block_mult(U_global, V_global, W_global,
                               0, 2, (2, 2),  # A1b
                               2, 3, (2, 3),  # B2b
                               0, 3, (2, 3),  # C12b
                               s2x2x3_U, s2x2x3_V, s2x2x3_W, next_col)
    
    # C21 = A2 * B1 (1x3) where A2 is row 2 of A (1x4)
    # A2a = A[2:3, 0:2], A2b = A[2:3, 2:4]
    # B1a = B[0:2, 0:3], B1b = B[2:4, 0:3]
    s1x2x3_U, s1x2x3_V, s1x2x3_W = schoolbook_1x2x3()
    next_col = add_block_mult(U_global, V_global, W_global,
                               2, 0, (1, 2),  # A2a
                               0, 0, (2, 3),  # B1a
                               2, 0, (1, 3),  # C21a: row 2, cols 0-2
                               s1x2x3_U, s1x2x3_V, s1x2x3_W, next_col)
    
    next_col = add_block_mult(U_global, V_global, W_global,
                               2, 2, (1, 2),  # A2b
                               2, 0, (2, 3),  # B1b
                               2, 0, (1, 3),  # C21b
                               s1x2x3_U, s1x2x3_V, s1x2x3_W, next_col)
    
    # C22 = A2 * B2 (1x3)
    next_col = add_block_mult(U_global, V_global, W_global,
                               2, 0, (1, 2),  # A2a
                               0, 3, (2, 3),  # B2a
                               2, 3, (1, 3),  # C22a: row 2, cols 3-5
                               s1x2x3_U, s1x2x3_V, s1x2x3_W, next_col)
    
    next_col = add_block_mult(U_global, V_global, W_global,
                               2, 2, (1, 2),  # A2b
                               2, 3, (2, 3),  # B2b
                               2, 3, (1, 3),  # C22b
                               s1x2x3_U, s1x2x3_V, s1x2x3_W, next_col)
    
    # Convert to lists with integer/half-integer entries
    U_list = [[int(U_global[i][j]) if U_global[i][j] == int(U_global[i][j]) else float(U_global[i][j]) for j in range(total_rank)] for i in range(n*m)]
    V_list = [[int(V_global[i][j]) if V_global[i][j] == int(V_global[i][j]) else float(V_global[i][j]) for j in range(total_rank)] for i in range(m*p)]
    W_list = [[int(W_global[i][j]) if W_global[i][j] == int(W_global[i][j]) else float(W_global[i][j]) for j in range(total_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
