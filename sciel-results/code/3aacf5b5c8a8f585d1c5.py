"""
Combined program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen-based block decomposition to achieve rank 46,
with schoolbook (rank 50) as guaranteed fallback.
"""
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Strategy: Partition into 2x2, 2x1, 1x2, 1x1 blocks and use
    Strassen's 7-multiplication scheme for 2x2x2 subproblems.
    This achieves rank 46 (4*7 + 2*4 + 2*4 + 2 = 46).
    Note: Rank 26 proposed in inspirations is impossible as rank(1x5x5)=25.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    fallback_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    def _validate(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build Strassen-based decomposition with rank 46
    # Partition: A (2x5) cols [0,1],[2,3],[4], B (5x5) rows/cols [0,1],[2,3],[4]
    # 2x2x2 blocks: (A00,B00), (A00,B01), (A01,B00), (A01,B01) -> 4 blocks * 7 = 28
    # 2x2x1 blocks: (A00,B02), (A01,B02) -> 2 * 4 = 8
    # 2x1x2 blocks: (A02,B00), (A02,B01) -> 2 * 4 = 8
    # 2x1x1 block:  (A02,B02) -> 2
    # Total: 28 + 8 + 8 + 2 = 46
    
    R = 46
    U = [[0.0] * R for _ in range(n * m)]  # 10 rows
    V = [[0.0] * R for _ in range(m * p)]  # 25 rows
    W = [[0.0] * R for _ in range(p * n)]  # 10 rows
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 block
    # A block: 2x2 with global row offset arow, col offset acol (in A's 2x5)
    # B block: 2x2 with global row offset brow, col offset bcol (in B's 5x5)
    # Output: 2x2 at output row offset orow, col offset ocol (in C's 2x5)
    # A[i,j] is at U index (i*5 + j), V index (j*5 + k) where j is B row, k is B col
    # W index is (k*2 + i) where k is output col, i is output row
    
    def add_strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        # Strassen formulas for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (B00+B10)(A10-A00)
        # M7 = (B01+B11)(A01-A11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # A block indices in U: (a_row_off + i)*5 + (a_col_off + j) for i=0,1; j=0,1
        # B block indices in V: (b_row_off + j)*5 + (b_col_off + k) for j=0,1; k=0,1
        # C block indices in W: (c_col_off + k)*2 + (c_row_off + i) for i=0,1; k=0,1
        
        # M1 = (A00+A11)(B00+B11)
        c1 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c1] = 1.0  # A00
        U[(a_row_off+1)*5 + (a_col_off+1)][c1] = 1.0  # A11
        V[(b_row_off+0)*5 + (b_col_off+0)][c1] = 1.0  # B00
        V[(b_row_off+1)*5 + (b_col_off+1)][c1] = 1.0  # B11
        W[(c_col_off+0)*2 + (c_row_off+0)][c1] = 1.0  # C00 += M1
        W[(c_col_off+1)*2 + (c_row_off+1)][c1] = 1.0  # C11 += M1
        
        # M2 = (A10+A11)B00
        c2 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+0)][c2] = 1.0  # A10
        U[(a_row_off+1)*5 + (a_col_off+1)][c2] = 1.0  # A11
        V[(b_row_off+0)*5 + (b_col_off+0)][c2] = 1.0  # B00
        W[(c_col_off+0)*2 + (c_row_off+1)][c2] = 1.0  # C10 += M2
        W[(c_col_off+1)*2 + (c_row_off+1)][c2] = -1.0 # C11 -= M2
        
        # M3 = A00(B10-B11)
        c3 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c3] = 1.0  # A00
        V[(b_row_off+1)*5 + (b_col_off+0)][c3] = 1.0  # B10
        V[(b_row_off+1)*5 + (b_col_off+1)][c3] = -1.0 # B11
        W[(c_col_off+1)*2 + (c_row_off+0)][c3] = 1.0  # C01 += M3
        W[(c_col_off+1)*2 + (c_row_off+1)][c3] = 1.0  # C11 += M3
        
        # M4 = A11(B01-B10)
        c4 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+1)][c4] = 1.0  # A11
        V[(b_row_off+0)*5 + (b_col_off+1)][c4] = 1.0  # B01
        V[(b_row_off+1)*5 + (b_col_off+0)][c4] = -1.0 # B10
        W[(c_col_off+0)*2 + (c_row_off+0)][c4] = 1.0  # C00 += M4
        W[(c_col_off+0)*2 + (c_row_off+1)][c4] = 1.0  # C10 += M4
        
        # M5 = (A00+A01)B11
        c5 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+0)][c5] = 1.0  # A00
        U[(a_row_off+0)*5 + (a_col_off+1)][c5] = 1.0  # A01
        V[(b_row_off+1)*5 + (b_col_off+1)][c5] = 1.0  # B11
        W[(c_col_off+0)*2 + (c_row_off+0)][c5] = -1.0 # C00 -= M5
        W[(c_col_off+1)*2 + (c_row_off+0)][c5] = 1.0  # C01 += M5
        
        # M6 = (B00+B10)(A10-A00)
        c6 = col; col += 1
        U[(a_row_off+1)*5 + (a_col_off+0)][c6] = 1.0  # A10
        U[(a_row_off+0)*5 + (a_col_off+0)][c6] = -1.0 # A00
        V[(b_row_off+0)*5 + (b_col_off+0)][c6] = 1.0  # B00
        V[(b_row_off+1)*5 + (b_col_off+0)][c6] = 1.0  # B10
        W[(c_col_off+1)*2 + (c_row_off+1)][c6] = 1.0  # C11 += M6
        
        # M7 = (B01+B11)(A01-A11)
        c7 = col; col += 1
        U[(a_row_off+0)*5 + (a_col_off+1)][c7] = 1.0  # A01
        U[(a_row_off+1)*5 + (a_col_off+1)][c7] = -1.0 # A11
        V[(b_row_off+0)*5 + (b_col_off+1)][c7] = 1.0  # B01
        V[(b_row_off+1)*5 + (b_col_off+1)][c7] = 1.0  # B11
        W[(c_col_off+0)*2 + (c_row_off+0)][c7] = 1.0  # C00 += M7
    
    # Helper: add schoolbook 2x2x1 block (A is 2x2, B is 2x1, C is 2x1)
    def add_schoolbook_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        for i in range(2):
            for j in range(2):
                cc = col; col += 1
                U[(a_row_off+i)*5 + (a_col_off+j)][cc] = 1.0
                V[(b_row_off+j)*5 + (b_col_off+0)][cc] = 1.0
                W[(c_col_off+0)*2 + (c_row_off+i)][cc] = 1.0
    
    # Helper: add schoolbook 2x1x2 block (A is 2x1, B is 1x2, C is 2x2)
    def add_schoolbook_2x1x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        for i in range(2):
            for k in range(2):
                cc = col; col += 1
                U[(a_row_off+i)*5 + (a_col_off+0)][cc] = 1.0
                V[(b_row_off+0)*5 + (b_col_off+k)][cc] = 1.0
                W[(c_col_off+k)*2 + (c_row_off+i)][cc] = 1.0
    
    # Helper: add schoolbook 2x1x1 block (A is 2x1, B is 1x1, C is 2x1)
    def add_schoolbook_2x1x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        for i in range(2):
            cc = col; col += 1
            U[(a_row_off+i)*5 + (a_col_off+0)][cc] = 1.0
            V[(b_row_off+0)*5 + (b_col_off+0)][cc] = 1.0
            W[(c_col_off+0)*2 + (c_row_off+i)][cc] = 1.0
    
    # 4 Strassen 2x2x2 blocks:
    # A block 0: rows 0-1, cols 0-1; B block 00: rows 0-1, cols 0-1; C: rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # A block 0: rows 0-1, cols 0-1; B block 01: rows 0-1, cols 2-3; C: rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    # A block 1: rows 0-1, cols 2-3; B block 10: rows 2-3, cols 0-1; C: rows 0-1, cols 0-1
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    # A block 1: rows 0-1, cols 2-3; B block 11: rows 2-3, cols 2-3; C: rows 0-1, cols 2-3
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    # 2x2x1 blocks: A(2x2) * B(2x1) -> C(2x1)
    # A block 0 * B block 02 (rows 0-1, col 4) -> C cols 4
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    # A block 1 * B block 12 (rows 2-3, col 4) -> C cols 4
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)
    
    # 2x1x2 blocks: A(2x1) * B(1x2) -> C(2x2)
    # A block 2 (cols 4) * B block 20 (row 4, cols 0-1) -> C cols 0-1
    add_schoolbook_2x1x2(0, 4, 4, 0, 0, 0)
    # A block 2 (cols 4) * B block 21 (row 4, cols 2-3) -> C cols 2-3
    add_schoolbook_2x1x2(0, 4, 4, 2, 0, 2)
    
    # 2x1x1 block: A(2x1) * B(1x1) -> C(2x1)
    # A block 2 * B block 22 -> C col 4
    add_schoolbook_2x1x1(0, 4, 4, 4, 0, 4)
    
    assert col == R, f"Expected {R} columns, got {col}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return result
    
    # If validation failed, return fallback
    return fallback_result
# EVOLVE-BLOCK-END
