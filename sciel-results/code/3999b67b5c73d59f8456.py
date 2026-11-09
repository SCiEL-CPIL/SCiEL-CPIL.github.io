"""Matrix multiplication 4x5x5 with block-Strassen decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Return (U, V, W) for M_{n,m,p} with U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """Return (U, V, W) for M_{2,2,2} Strassen, rank 7."""
    # U rows: 0:A11, 1:A12, 2:A21, 3:A22
    # V rows: 0:B11, 1:B12, 2:B21, 3:B22
    # W rows: 0:C11, 1:C12, 2:C21, 3:C22
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Split A (4x5) into rows [0:2], [2:4] and cols [0:2], [2:5]
    # Split B (5x5) into rows [0:2], [2:5] and cols [0:2], [2:5]
    # C is 4x5, rows [0:2], [2:4], cols [0:2], [2:5]
    
    # Block sizes:
    # A11: 2x2, A12: 2x3, A21: 2x2, A22: 2x3
    # B11: 2x2, B12: 2x3, B21: 3x2, B22: 3x3
    # C11: 2x2, C12: 2x3, C21: 2x2, C22: 2x3
    
    # We need to build U (20 x R), V (25 x R), W (20 x R)
    # U rows: A11 rows 0-1 (cols 0-1), A12 rows 0-1 (cols 2-4), A21 rows 2-3 (cols 0-1), A22 rows 2-3 (cols 2-4)
    # V rows: B11 rows 0-1 (cols 0-1), B12 rows 0-1 (cols 2-4), B21 rows 2-4 (cols 0-1), B22 rows 2-4 (cols 2-4)
    # W rows: C11 rows 0-1 (cols 0-1), C12 rows 0-1 (cols 2-4), C21 rows 2-3 (cols 0-1), C22 rows 2-3 (cols 2-4)
    
    # Total rank: sum of ranks of each block multiplication
    # Block 1: A11(2x2) * B11(2x2) -> C11(2x2): Strassen rank 7
    # Block 2: A11(2x2) * B12(2x3) -> C12(2x3): Schoolbook rank 12
    # Block 3: A12(2x3) * B11(2x2) -> C12(2x3): Schoolbook rank 12
    # Block 4: A12(2x3) * B12(2x3) -> C12(2x3): Schoolbook rank 18
    # Block 5: A21(2x2) * B21(3x2) -> C21(2x2): Schoolbook rank 12
    # Block 6: A21(2x2) * B22(3x3) -> C21(2x2): Schoolbook rank 12
    # Block 7: A22(2x3) * B21(3x2) -> C22(2x3): Schoolbook rank 18
    # Block 8: A22(2x3) * B22(3x3) -> C22(2x3): Schoolbook rank 18
    
    # Wait, let me recompute. The block multiplication for C11 is A11*B11.
    # C12 = A11*B12 + A12*B12? No. C = A*B.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C11 (rows 0-1, cols 0-1) = A11(0-1,0-1)*B11(0-1,0-1) + A12(0-1,2-4)*B21(2-4,0-1)
    # C12 (rows 0-1, cols 2-4) = A11(0-1,0-1)*B12(0-1,2-4) + A12(0-1,2-4)*B22(2-4,2-4)
    # C21 (rows 2-3, cols 0-1) = A21(2-3,0-1)*B11(0-1,0-1) + A22(2-3,2-4)*B21(2-4,0-1)
    # C22 (rows 2-3, cols 2-4) = A21(2-3,0-1)*B12(0-1,2-4) + A22(2-3,2-4)*B22(2-4,2-4)
    
    # So we have 8 block multiplications:
    # 1. A11(2x2) * B11(2x2) -> contributes to C11(2x2): Strassen rank 7
    # 2. A12(2x3) * B21(3x2) -> contributes to C11(2x2): Schoolbook rank 12
    # 3. A11(2x2) * B12(2x3) -> contributes to C12(2x3): Schoolbook rank 12
    # 4. A12(2x3) * B22(3x3) -> contributes to C12(2x3): Schoolbook rank 18
    # 5. A21(2x2) * B11(2x2) -> contributes to C21(2x2): Strassen rank 7
    # 6. A22(2x3) * B21(3x2) -> contributes to C21(2x2): Schoolbook rank 12
    # 7. A21(2x2) * B12(2x3) -> contributes to C22(2x3): Schoolbook rank 12
    # 8. A22(2x3) * B22(3x3) -> contributes to C22(2x3): Schoolbook rank 18
    
    total_rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18
    # = 98
    
    U = np.zeros((n * m, total_rank), dtype=int)  # 20 x 98
    V = np.zeros((m * p, total_rank), dtype=int)  # 25 x 98
    W = np.zeros((p * n, total_rank), dtype=int)  # 20 x 98
    
    col_offset = 0
    
    # Helper to add a block decomposition
    def add_block(a_row_start, a_col_start, a_rows, a_cols,
                  b_row_start, b_col_start, b_rows, b_cols,
                  c_row_start, c_col_start, c_rows, c_cols,
                  use_strassen=False):
        nonlocal col_offset
        if use_strassen:
            bu, bv, bw = _strassen_222()
        else:
            bu, bv, bw = _schoolbook(a_rows, a_cols, b_cols)
        r = bu.shape[1]
        # Map U rows: a_row_start + i, a_col_start + j -> U[(a_row_start+i)*m + (a_col_start+j)]
        # bu rows are indexed by (i, j) in block coords, row = i*a_cols + j
        for ridx in range(bu.shape[0]):
            i, j = divmod(ridx, a_cols)
            global_row = (a_row_start + i) * m + (a_col_start + j)
            U[global_row, col_offset:col_offset+r] += bu[ridx, :]
        # Map V rows: b_row_start + i, b_col_start + j -> V[(b_row_start+i)*p + (b_col_start+j)]
        # bv rows are indexed by (i, j) in block coords, row = i*b_cols + j
        for ridx in range(bv.shape[0]):
            i, j = divmod(ridx, b_cols)
            global_row = (b_row_start + i) * p + (b_col_start + j)
            V[global_row, col_offset:col_offset+r] += bv[ridx, :]
        # Map W rows: c_row_start + i, c_col_start + j -> W[(c_col_start+j)*n + (c_row_start+i)]
        # bw rows are indexed by (i, j) in block coords, row = i*c_cols + j
        for ridx in range(bw.shape[0]):
            i, j = divmod(ridx, c_cols)
            global_row = (c_col_start + j) * n + (c_row_start + i)
            W[global_row, col_offset:col_offset+r] += bw[ridx, :]
        col_offset += r
    
    # Block 1: A11(0,0,2,2) * B11(0,0,2,2) -> C11(0,0,2,2) Strassen
    add_block(0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, use_strassen=True)
    # Block 2: A12(0,2,2,3) * B21(2,0,3,2) -> C11(0,0,2,2) Schoolbook
    add_block(0, 2, 2, 3, 2, 0, 3, 2, 0, 0, 2, 2)
    # Block 3: A11(0,0,2,2) * B12(0,2,2,3) -> C12(0,2,2,3) Schoolbook
    add_block(0, 0, 2, 2, 0, 2, 2, 3, 0, 2, 2, 3)
    # Block 4: A12(0,2,2,3) * B22(2,2,3,3) -> C12(0,2,2,3) Schoolbook
    add_block(0, 2, 2, 3, 2, 2, 3, 3, 0, 2, 2, 3)
    # Block 5: A21(2,0,2,2) * B11(0,0,2,2) -> C21(2,0,2,2) Strassen
    add_block(2, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, use_strassen=True)
    # Block 6: A22(2,2,2,3) * B21(2,0,3,2) -> C21(2,0,2,2) Schoolbook
    add_block(2, 2, 2, 3, 2, 0, 3, 2, 2, 0, 2, 2)
    # Block 7: A21(2,0,2,2) * B12(0,2,2,3) -> C22(2,2,2,3) Schoolbook
    add_block(2, 0, 2, 2, 0, 2, 2, 3, 2, 2, 2, 3)
    # Block 8: A22(2,2,2,3) * B22(2,2,3,3) -> C22(2,2,2,3) Schoolbook
    add_block(2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 3)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
