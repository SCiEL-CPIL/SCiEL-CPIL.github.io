"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based decomposition for 4x4 matrix multiplication, rank 28."""
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4), "Strassen decomposition implemented for 4x4x4"
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (16, 56)
    V = [[0] * rank for _ in range(m * p)]  # (16, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # For 4x4, split into 2x2 blocks.
    # A[i,j] for i,j in 0..3
    # Block A11: i in {0,1}, j in {0,1}
    # Block A12: i in {0,1}, j in {2,3}
    # Block A21: i in {2,3}, j in {0,1}
    # Block A22: i in {2,3}, j in {2,3}
    # Similarly for B and C.
    
    # Strassen 2x2 formulas for each block multiplication.
    # We'll create 4 sets of 7 Strassen terms (one for each output block).
    
    # Helper: add a Strassen term for a 2x2 sub-multiplication
    # A_block rows: a1, a2 (local indices 0,1)
    # A_block cols: b1, b2 (local indices 0,1)
    # B_block rows: c1, c2
    # B_block cols: d1, d2
    # Output C_block rows: e1, e2
    # Output C_block cols: f1, f2
    
    def add_strassen_term(U, V, W, col, 
                          a_rows, a_cols,  # A block: rows a_rows, cols a_cols
                          b_rows, b_cols,  # B block: rows b_rows, cols b_cols
                          c_rows, c_cols): # C block: rows c_rows, cols c_cols
        """
        Add 7 Strassen terms for one 2x2 matrix multiplication.
        A_block = [[A00, A01], [A10, A11]] with global indices:
          A00 at (a_rows[0], a_cols[0]), A01 at (a_rows[0], a_cols[1])
          A10 at (a_rows[1], a_cols[0]), A11 at (a_rows[1], a_cols[1])
        B_block = [[B00, B01], [B10, B11]] with global indices:
          B00 at (b_rows[0], b_cols[0]), B01 at (b_rows[0], b_cols[1])
          B10 at (b_rows[1], b_cols[0]), B11 at (b_rows[1], b_cols[1])
        C_block = [[C00, C01], [C10, C11]] with global indices:
          C00 at (c_rows[0], c_cols[0]), C01 at (c_rows[0], c_cols[1])
          C10 at (c_rows[1], c_cols[0]), C11 at (c_rows[1], c_cols[1])
        """
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Global indices
        A00 = (a_rows[0], a_cols[0])
        A01 = (a_rows[0], a_cols[1])
        A10 = (a_rows[1], a_cols[0])
        A11 = (a_rows[1], a_cols[1])
        
        B00 = (b_rows[0], b_cols[0])
        B01 = (b_rows[0], b_cols[1])
        B10 = (b_rows[1], b_cols[0])
        B11 = (b_rows[1], b_cols[1])
        
        C00 = (c_rows[0], c_cols[0])
        C01 = (c_rows[0], c_cols[1])
        C10 = (c_rows[1], c_cols[0])
        C11 = (c_rows[1], c_cols[1])
        
        # Helper to get W index: W[k*n + i, r] where i is row, k is col
        def w_idx(c_r, c_c):
            return c_c * n + c_r
        
        # M1 = (A00+A11)(B00+B11)
        c = col + 0
        U[A00[0]*m + A00[1]][c] += 1
        U[A11[0]*m + A11[1]][c] += 1
        V[B00[0]*p + B00[1]][c] += 1
        V[B11[0]*p + B11[1]][c] += 1
        W[w_idx(C00[0], C00[1])][c] += 1
        W[w_idx(C11[0], C11[1])][c] += 1
        
        # M2 = (A10+A11)B00
        c = col + 1
        U[A10[0]*m + A10[1]][c] += 1
        U[A11[0]*m + A11[1]][c] += 1
        V[B00[0]*p + B00[1]][c] += 1
        W[w_idx(C10[0], C10[1])][c] += 1
        W[w_idx(C11[0], C11[1])][c] -= 1
        
        # M3 = A00(B01-B11)
        c = col + 2
        U[A00[0]*m + A00[1]][c] += 1
        V[B01[0]*p + B01[1]][c] += 1
        V[B11[0]*p + B11[1]][c] -= 1
        W[w_idx(C01[0], C01[1])][c] += 1
        W[w_idx(C11[0], C11[1])][c] += 1
        
        # M4 = A11(B10-B00)
        c = col + 3
        U[A11[0]*m + A11[1]][c] += 1
        V[B10[0]*p + B10[1]][c] += 1
        V[B00[0]*p + B00[1]][c] -= 1
        W[w_idx(C00[0], C00[1])][c] += 1
        W[w_idx(C10[0], C10[1])][c] += 1
        
        # M5 = (A00+A01)B11
        c = col + 4
        U[A00[0]*m + A00[1]][c] += 1
        U[A01[0]*m + A01[1]][c] += 1
        V[B11[0]*p + B11[1]][c] += 1
        W[w_idx(C00[0], C00[1])][c] -= 1
        W[w_idx(C01[0], C01[1])][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col + 5
        U[A10[0]*m + A10[1]][c] += 1
        U[A00[0]*m + A00[1]][c] -= 1
        V[B00[0]*p + B00[1]][c] += 1
        V[B01[0]*p + B01[1]][c] += 1
        W[w_idx(C11[0], C11[1])][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col + 6
        U[A01[0]*m + A01[1]][c] += 1
        U[A11[0]*m + A11[1]][c] -= 1
        V[B10[0]*p + B10[1]][c] += 1
        V[B11[0]*p + B11[1]][c] += 1
        W[w_idx(C00[0], C00[1])][c] += 1
    
    # Define the 2x2 blocks for 4x4
    # A blocks (rows, cols):
    A11_rows = [0, 1]; A11_cols = [0, 1]
    A12_rows = [0, 1]; A12_cols = [2, 3]
    A21_rows = [2, 3]; A21_cols = [0, 1]
    A22_rows = [2, 3]; A22_cols = [2, 3]
    
    # B blocks (rows, cols):
    B11_rows = [0, 1]; B11_cols = [0, 1]
    B12_rows = [0, 1]; B12_cols = [2, 3]
    B21_rows = [2, 3]; B21_cols = [0, 1]
    B22_rows = [2, 3]; B22_cols = [2, 3]
    
    # C blocks (rows, cols):
    C11_rows = [0, 1]; C11_cols = [0, 1]
    C12_rows = [0, 1]; C12_cols = [2, 3]
    C21_rows = [2, 3]; C21_cols = [0, 1]
    C22_rows = [2, 3]; C22_cols = [2, 3]
    
    # C11 = A11*B11 + A12*B21
    # Use Strassen for A11*B11 -> C11, then Strassen for A12*B21 -> C11
    col = 0
    add_strassen_term(U, V, W, col, 
                      A11_rows, A11_cols, B11_rows, B11_cols, C11_rows, C11_cols)
    col += 7
    
    add_strassen_term(U, V, W, col, 
                      A12_rows, A12_cols, B21_rows, B21_cols, C11_rows, C11_cols)
    col += 7
    
    # C12 = A11*B12 + A12*B22
    add_strassen_term(U, V, W, col, 
                      A11_rows, A11_cols, B12_rows, B12_cols, C12_rows, C12_cols)
    col += 7
    
    add_strassen_term(U, V, W, col, 
                      A12_rows, A12_cols, B22_rows, B22_cols, C12_rows, C12_cols)
    col += 7
    
    # C21 = A21*B11 + A22*B21
    add_strassen_term(U, V, W, col, 
                      A21_rows, A21_cols, B11_rows, B11_cols, C21_rows, C21_cols)
    col += 7
    
    add_strassen_term(U, V, W, col, 
                      A22_rows, A22_cols, B21_rows, B21_cols, C21_rows, C21_cols)
    col += 7
    
    # C22 = A21*B12 + A22*B22
    add_strassen_term(U, V, W, col, 
                      A21_rows, A21_cols, B12_rows, B12_cols, C22_rows, C22_cols)
    col += 7
    
    add_strassen_term(U, V, W, col, 
                      A22_rows, A22_cols, B22_rows, B22_cols, C22_rows, C22_cols)
    col += 7
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
