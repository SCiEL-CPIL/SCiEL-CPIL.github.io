import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This should equal 1 for all i,j,k
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _strassen_44():
    """Rank-44 decomposition: 2x4x4 via Strassen on 2x2 blocks (rank 28) + 1x4x4 naive (rank 16)."""
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    # Strassen 2x2x2 steps: (u_coeffs, v_coeffs, w_coeffs)
    # Each maps (li,lj)->val for U, (lj,lk)->val for V, (lk,li)->val for W
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (X10 + X11)Y00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00(Y01 - Y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11(Y10 - Y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00 + X01)Y11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Four 2x2 blocks for the 2x4 part of A (rows 0,1) and 4x4 B
    # Block (row_block, col_block) where row_block in {0,1} (for A's 2 rows), col_block in {0,1}
    # A[rows 2*row_block : 2*row_block+2, cols 2*col_block : 2*col_block+2] * B[cols 2*col_block : 2*col_block+2, all 4 cols]
    # Output: C[rows 2*row_block : 2*row_block+2, all 4 cols]
    
    col = 0
    for row_block in range(2):  # rows 0-1 of A
        for col_block in range(2):  # cols 0-3 of A split into two 2-col groups
            for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
                for (li, lj), val in u_coeffs.items():
                    U[(li + row_block * 2) * m + (lj + col_block * 2)][col] += val
                for (lj, lk), val in v_coeffs.items():
                    V[(lj + col_block * 2) * p + lk][col] += val
                for (lk, li), val in w_coeffs.items():
                    W[lk * n + (li + row_block * 2)][col] += val
                col += 1
    
    # 1x4x4 naive for row 2 of A
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank 44 Strassen decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if ok:
        return result_44
    
    # Fallback to schoolbook
    return _schoolbook()
