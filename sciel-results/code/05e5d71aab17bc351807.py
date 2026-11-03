import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    deadline = time.time() + 145.0
    
    # Rank-7 bilinear algorithm for 2x2 matrix multiplication
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X @ Y, Z[i,k]
    #
    # 7 products:
    # M0 = (x00 + x11) * (y00 + y11)
    # M1 = (x01 + x11) * y00
    # M2 = x00 * (y10 - y00)
    # M3 = x11 * (y01 - y11)
    # M4 = (x00 + x01) * y11
    # M5 = (x10 - x00) * (y00 + y01)
    # M6 = (x01 + x10) * (y10 + y11)
    #
    # Reconstruction:
    # Z00 = M0 + M2 - M4 - M5
    # Z01 = M3 + M4 - M5
    # Z10 = M1 + M2 - M4 + M6
    # Z11 = M0 - M1 + M3 + M6
    
    # U22: 4 rows (x00,x01,x10,x11) x 7 cols
    U22 = [
        [1, 0, 1, 0, 1, -1, 0],  # x00
        [0, 1, 0, 0, 1, 0, 1],   # x01
        [0, 0, 0, 0, 0, 1, 1],   # x10
        [1, 1, 0, 1, 0, 0, 0],   # x11
    ]
    
    # V22: 4 rows (y00,y01,y10,y11) x 7 cols
    V22 = [
        [1, 1, -1, 0, 0, 1, 0],  # y00
        [1, 0, 0, 1, 0, 1, 0],   # y01
        [1, 0, 1, 0, 0, 0, 1],   # y10
        [1, 0, 0, -1, 1, 0, 1],  # y11
    ]
    
    # W22: 4 rows (Z00,Z01,Z10,Z11) x 7 cols
    W22 = [
        [1, 0, 1, 0, -1, -1, 0],  # Z00
        [0, 0, 0, 1, 1, -1, 0],   # Z01
        [0, 1, 1, 0, -1, 0, 1],   # Z10
        [1, -1, 0, 1, 0, 0, 1],   # Z11
    ]
    
    # Verify the 2x2 decomposition
    U22_arr = np.array(U22, dtype=float)
    V22_arr = np.array(V22, dtype=float)
    W22_arr = np.array(W22, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    X_test = np.random.randn(2, 2)
    Y_test = np.random.randn(2, 2)
    Z_expected = X_test @ Y_test
    
    # Flatten X and Y
    X_flat = X_test.flatten()  # [x00, x01, x10, x11]
    Y_flat = Y_test.flatten()  # [y00, y01, y10, y11]
    
    # Compute via decomposition
    products = U22_arr @ X_flat * (V22_arr @ Y_flat)
    Z_computed = W22_arr @ products
    
    if np.max(np.abs(Z_computed - Z_expected)) > 1e-10:
        # Fallback to schoolbook if 2x2 is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x4 x 4x6 decomposition
    # Split A (2x4) into two 2x2 column blocks: A = [A_left | A_right]
    # Split B (4x6) into two 2x6 row blocks: B = [B_top; B_bottom]
    # C = A_left @ B_top + A_right @ B_bottom
    # Each term: 2x2 @ 2x6
    # Split 2x6 into three 2x2 column blocks
    # Total: 2 * 3 = 6 blocks of 2x2 @ 2x2, each rank 7
    # Total rank: 6 * 7 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Blocks: (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    # A is 2x4, B is 4x6
    # A_left: rows 0-1, cols 0-1
    # A_right: rows 0-1, cols 2-3
    # B_top: rows 0-1, cols 0-5
    # B_bottom: rows 2-3, cols 0-5
    
    blocks = [
        # (a_row_start, a_col_start, b_row_start, b_col_start)
        (0, 0, 0, 0),    # A_left, B_top, col block 0
        (0, 0, 0, 2),    # A_left, B_top, col block 1
        (0, 0, 0, 4),    # A_left, B_top, col block 2
        (0, 2, 2, 0),    # A_right, B_bottom, col block 0
        (0, 2, 2, 2),    # A_right, B_bottom, col block 1
        (0, 2, 2, 4),    # A_right, B_bottom, col block 2
    ]
    
    col = 0
    for a_r, a_c, b_r, b_c in blocks:
        if time.time() > deadline:
            break
        
        # U: A indices. A is 2x4, row a_r+i, col a_c+j for i,j in {0,1}
        # Local 2x2: x00=A[a_r, a_c], x01=A[a_r, a_c+1], x10=A[a_r+1, a_c], x11=A[a_r+1, a_c+1]
        # U row index: i*m + j where i is row, j is col in A (2x4)
        a_indices = [
            a_r * m + a_c,       # x00
            a_r * m + (a_c + 1), # x01
            (a_r + 1) * m + a_c,       # x10
            (a_r + 1) * m + (a_c + 1), # x11
        ]
        
        # V: B indices. B is 4x6, row b_r+i, col b_c+j for i,j in {0,1}
        # Local 2x2: y00=B[b_r, b_c], y01=B[b_r, b_c+1], y10=B[b_r+1, b_c], y11=B[b_r+1, b_c+1]
        # V row index: i*p + j where i is row, j is col in B (4x6)
        b_indices = [
            b_r * p + b_c,       # y00
            b_r * p + (b_c + 1), # y01
            (b_r + 1) * p + b_c,       # y10
            (b_r + 1) * p + (b_c + 1), # y11
        ]
        
        # W: C indices. C is 2x6, row a_r+i (since A has 2 rows), col b_c+j
        # W row index: k*n + i where k is col of C, i is row of C
        # C[i, k] = sum over j of A[i,j] * B[j,k]
        # For this block: C row is a_r+i (i in {0,1}), col is b_c+j (j in {0,1})
        # W row index: (b_c+j) * n + (a_r+i)
        c_indices = [
            (b_c + 0) * n + (a_r + 0),   # Z00: col b_c, row a_r
            (b_c + 1) * n + (a_r + 0),   # Z01: col b_c+1, row a_r
            (b_c + 0) * n + (a_r + 1),   # Z10: col b_c, row a_r+1
            (b_c + 1) * n + (a_r + 1),   # Z11: col b_c+1, row a_r+1
        ]
        
        for c in range(7):
            if time.time() > deadline:
                break
            for local_idx in range(4):
                U[a_indices[local_idx], col] += U22[local_idx][c]
                V[b_indices[local_idx], col] += V22[local_idx][c]
                W[c_indices[local_idx], col] += W22[local_idx][c]
            col += 1
    
    # Verify the full decomposition
    np.random.seed(123)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    A_flat = A_test.flatten()  # n*m = 8 elements
    B_flat = B_test.flatten()  # m*p = 24 elements
    
    products = (U @ A_flat) * (V @ B_flat)
    C_computed = W @ products
    
    if np.max(np.abs(C_computed - C_expected)) > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists with integer values (all entries are integers in this decomposition)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
