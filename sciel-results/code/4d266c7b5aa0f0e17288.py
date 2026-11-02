"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col
        # Strassen coefficients for U, V, W
        # U is (a_row_start + i) * m + (a_col_start + j)
        # V is (b_row_start + j) * p + (b_col_start + k)
        # W is (c_col_start + k) * n + (c_row_start + i)
        
        # M1 = (A00+A11)(B00+B11)
        U[a_row_start * m + a_col_start][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[b_row_start * p + b_col_start][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[c_col_start * n + c_row_start][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(a_row_start+1) * m + a_col_start][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[b_row_start * p + b_col_start][col] += 1
        W[c_col_start * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[a_row_start * m + a_col_start][col] += 1
        V[b_row_start * p + (b_col_start+1)][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] -= 1
        W[(c_col_start+1) * n + c_row_start][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(a_row_start+1) * m + (a_col_start+1)][col] += 1
        V[(b_row_start+1) * p + b_col_start][col] += 1
        V[b_row_start * p + b_col_start][col] -= 1
        W[c_col_start * n + (c_row_start+1)][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a_row_start * m + a_col_start][col] += 1
        U[a_row_start * m + (a_col_start+1)][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[c_col_start * n + c_row_start][col] -= 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(a_row_start+1) * m + a_col_start][col] += 1
        U[a_row_start * m + a_col_start][col] -= 1
        V[b_row_start * p + b_col_start][col] += 1
        V[b_row_start * p + (b_col_start+1)][col] += 1
        W[c_col_start * n + (c_row_start+1)][col] += 1
        W[(c_col_start+1) * n + (c_row_start+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_row_start * m + (a_col_start+1)][col] += 1
        U[(a_row_start+1) * m + (a_col_start+1)][col] -= 1
        V[(b_row_start+1) * p + b_col_start][col] += 1
        V[(b_row_start+1) * p + (b_col_start+1)][col] += 1
        W[c_col_start * n + c_row_start][col] += 1
        W[(c_col_start+1) * n + c_row_start][col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 0, 0, 4, 0, 4)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
