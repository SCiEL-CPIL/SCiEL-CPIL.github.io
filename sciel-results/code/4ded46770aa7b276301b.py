"""Matrix multiplication 2x4 by 4x4 using Strassen for 2x2 sub-blocks."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # We'll use Strassen for four 2x2 sub-problems
    # Total rank = 4 * 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Partition:
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i, k] where i in {0,1}, k in {0,1,2,3}
    #
    # Left half of C (k in {0,1}): C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
    #                             C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
    # Right half of C (k in {2,3}): C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2] + A[i,3]*B[3,2]
    #                              C[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3] + A[i,2]*B[2,3] + A[i,3]*B[3,3]
    #
    # Group A columns: A_left = [A[i,0], A[i,1]], A_right = [A[i,2], A[i,3]]
    # Group B rows: B_top = rows 0,1; B_bottom = rows 2,3
    # Group B columns: B_left = cols 0,1; B_right = cols 2,3
    #
    # C_left (2x2) = A_left * B_top_left + A_right * B_bottom_left
    # C_right (2x2) = A_left * B_top_right + A_right * B_bottom_right
    #
    # So we have 4 independent 2x2 multiplications:
    # 1. A_left * B_top_left -> contributes to C_left
    # 2. A_right * B_bottom_left -> contributes to C_left
    # 3. A_left * B_top_right -> contributes to C_right
    # 4. A_right * B_bottom_right -> contributes to C_right
    #
    # For each 2x2 multiplication, use Strassen (7 terms)
    
    def add_strassen_term(term_idx, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add one Strassen term for a 2x2 sub-problem.
        a_row_offset: row offset in A (0 or 1)
        a_col_offset: column offset in A (0 or 2)
        b_row_offset: row offset in B (0 or 2)
        b_col_offset: column offset in B (0 or 2)
        c_row_offset: row offset in C (0 or 1)
        c_col_offset: column offset in C (0 or 2)
        """
        # Strassen's 7 products for X (2x2) * Y (2x2):
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y01-y11)
        # M4 = x11(y10-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        # Result:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # For each Strassen term, we set U, V, W entries
        # U[i*m+j, r] where i is A row, j is A column
        # V[j*p+k, r] where j is B row, k is B column
        # W[k*n+i, r] where k is C column, i is C row
        
        # Term 1: M1 = (x00+x11)(y00+y11)
        r = term_idx * 7 + 0
        # U: A[x00] and A[x11] have coefficient +1
        U[a_row_offset * m + a_col_offset][r] = 1       # x00
        U[(a_row_offset+1) * m + a_col_offset + 1][r] = 1  # x11
        # V: B[y00] and B[y11] have coefficient +1
        V[b_row_offset * p + b_col_offset][r] = 1       # y00
        V[(b_row_offset+1) * p + b_col_offset + 1][r] = 1  # y11
        # W: contributes to c00 (+1), c11 (+1)
        W[b_col_offset * n + c_row_offset][r] = 1       # c00
        W[(b_col_offset+1) * n + c_row_offset + 1][r] = 1  # c11
        
        # Term 2: M2 = (x10+x11)y00
        r = term_idx * 7 + 1
        U[(a_row_offset+1) * m + a_col_offset][r] = 1       # x10
        U[(a_row_offset+1) * m + a_col_offset + 1][r] = 1   # x11
        V[b_row_offset * p + b_col_offset][r] = 1           # y00
        W[b_col_offset * n + c_row_offset + 1][r] = 1       # c10
        
        # Term 3: M3 = x00(y01-y11)
        r = term_idx * 7 + 2
        U[a_row_offset * m + a_col_offset][r] = 1           # x00
        V[b_row_offset * p + b_col_offset + 1][r] = 1       # y01
        V[(b_row_offset+1) * p + b_col_offset + 1][r] = -1  # y11
        W[(b_col_offset+1) * n + c_row_offset][r] = 1       # c01
        W[(b_col_offset+1) * n + c_row_offset + 1][r] = 1   # c11
        
        # Term 4: M4 = x11(y10-y00)
        r = term_idx * 7 + 3
        U[(a_row_offset+1) * m + a_col_offset + 1][r] = 1   # x11
        V[(b_row_offset+1) * p + b_col_offset][r] = 1       # y10
        V[b_row_offset * p + b_col_offset][r] = -1           # y00
        W[b_col_offset * n + c_row_offset][r] = 1           # c00
        W[b_col_offset * n + c_row_offset + 1][r] = 1       # c10
        
        # Term 5: M5 = (x00+x01)y11
        r = term_idx * 7 + 4
        U[a_row_offset * m + a_col_offset][r] = 1           # x00
        U[a_row_offset * m + a_col_offset + 1][r] = 1       # x01
        V[(b_row_offset+1) * p + b_col_offset + 1][r] = 1   # y11
        W[b_col_offset * n + c_row_offset][r] = -1           # c00 (negative)
        W[(b_col_offset+1) * n + c_row_offset][r] = 1       # c01
        
        # Term 6: M6 = (x10-x00)(y00+y01)
        r = term_idx * 7 + 5
        U[(a_row_offset+1) * m + a_col_offset][r] = 1       # x10
        U[a_row_offset * m + a_col_offset][r] = -1           # x00
        V[b_row_offset * p + b_col_offset][r] = 1           # y00
        V[b_row_offset * p + b_col_offset + 1][r] = 1       # y01
        W[(b_col_offset+1) * n + c_row_offset + 1][r] = 1   # c11
        
        # Term 7: M7 = (x01-x11)(y10+y11)
        r = term_idx * 7 + 6
        U[a_row_offset * m + a_col_offset + 1][r] = 1       # x01
        U[(a_row_offset+1) * m + a_col_offset + 1][r] = -1   # x11
        V[(b_row_offset+1) * p + b_col_offset][r] = 1       # y10
        V[(b_row_offset+1) * p + b_col_offset + 1][r] = 1   # y11
        W[b_col_offset * n + c_row_offset][r] = 1           # c00
    
    # Sub-problem 1: A_left (cols 0,1) * B_top_left (rows 0,1, cols 0,1) -> C_left (rows 0,1, cols 0,1)
    add_strassen_term(0, a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=0, c_row_offset=0, c_col_offset=0)
    
    # Sub-problem 2: A_right (cols 2,3) * B_bottom_left (rows 2,3, cols 0,1) -> C_left (rows 0,1, cols 0,1)
    add_strassen_term(1, a_row_offset=0, a_col_offset=2, b_row_offset=2, b_col_offset=0, c_row_offset=0, c_col_offset=0)
    
    # Sub-problem 3: A_left (cols 0,1) * B_top_right (rows 0,1, cols 2,3) -> C_right (rows 0,1, cols 2,3)
    add_strassen_term(2, a_row_offset=0, a_col_offset=0, b_row_offset=0, b_col_offset=2, c_row_offset=0, c_col_offset=2)
    
    # Sub-problem 4: A_right (cols 2,3) * B_bottom_right (rows 2,3, cols 2,3) -> C_right (rows 0,1, cols 2,3)
    add_strassen_term(3, a_row_offset=0, a_col_offset=2, b_row_offset=2, b_col_offset=2, c_row_offset=0, c_col_offset=2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
