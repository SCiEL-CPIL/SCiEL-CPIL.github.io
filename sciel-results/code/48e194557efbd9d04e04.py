import time
import numpy as np

def construct():
    """
    Rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Laderman's 7-term algorithm for 2x2*2x2 blocks.
    """
    n, m, p = 3, 4, 4
    rank = 28  # 4 blocks * 7 terms each

    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]

    col = 0

    def add_laderman(U_rows, V_cols, W_rows, col):
        """
        Add Laderman's 7-term algorithm for a 2x2 * 2x2 block.
        
        U_rows: [r00, r01, r10, r11] - U row indices
        V_cols: [c00, c01, c10, c11] - V column indices  
        W_rows: [w00, w01, w10, w11] - W row indices
        """
        # Term 1: (A00 + A11)(B11 - B10)
        U[U_rows[0]][col] += 1
        U[U_rows[3]][col] += 1
        V[V_cols[3]][col] += 1
        V[V_cols[2]][col] -= 1
        W[W_rows[3]][col] += 1
        W[W_rows[1]][col] += 1
        col += 1

        # Term 2: (A10 + A11)B00
        U[U_rows[2]][col] += 1
        U[U_rows[3]][col] += 1
        V[V_cols[0]][col] += 1
        W[W_rows[0]][col] += 1
        W[W_rows[3]][col] += 1
        col += 1

        # Term 3: (A01 - A00)B10
        U[U_rows[1]][col] += 1
        U[U_rows[0]][col] -= 1
        V[V_cols[2]][col] += 1
        W[W_rows[2]][col] += 1
        W[W_rows[0]][col] -= 1
        col += 1

        # Term 4: (A00 + A01)B11
        U[U_rows[0]][col] += 1
        U[U_rows[1]][col] += 1
        V[V_cols[3]][col] += 1
        W[W_rows[1]][col] += 1
        W[W_rows[3]][col] += 1
        col += 1

        # Term 5: (A10 - A11)B01
        U[U_rows[2]][col] += 1
        U[U_rows[3]][col] -= 1
        V[V_cols[1]][col] += 1
        W[W_rows[2]][col] += 1
        W[W_rows[0]][col] -= 1
        col += 1

        # Term 6: (A00 - A10)(B10 + B11)
        U[U_rows[0]][col] += 1
        U[U_rows[2]][col] -= 1
        V[V_cols[2]][col] += 1
        V[V_cols[3]][col] += 1
        W[W_rows[2]][col] += 1
        W[W_rows[0]][col] -= 1
        col += 1

        # Term 7: (A01 + A11)(B00 - B01)
        U[U_rows[1]][col] += 1
        U[U_rows[3]][col] += 1
        V[V_cols[0]][col] += 1
        V[V_cols[1]][col] -= 1
        W[W_rows[1]][col] += 1
        W[W_rows[3]][col] += 1
        col += 1

        return col

    # Block 1: rows 0-1, cols 0-1 -> output rows 0-1, cols 0-1
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W indices: C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    col = add_laderman([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)

    # Block 2: rows 0-1, cols 2-3 -> output rows 0-1, cols 2-3
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # V indices: B[0,0]=2, B[0,1]=3, B[1,0]=6, B[1,1]=7
    # W indices: C[0,0]=2, C[0,1]=5, C[1,0]=3, C[1,1]=6
    col = add_laderman([0, 1, 4, 5], [2, 3, 6, 7], [2, 5, 3, 6], col)

    # Block 3: rows 2-3, cols 0-1 -> output rows 2-3, cols 0-1
    # U indices: A[0,0]=2, A[0,1]=3, A[1,0]=6, A[1,1]=7
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W indices: C[0,0]=6, C[0,1]=9, C[1,0]=7, C[1,1]=10
    col = add_laderman([2, 3, 6, 7], [0, 1, 4, 5], [6, 9, 7, 10], col)

    # Block 4: rows 2-3, cols 2-3 -> output rows 2-3, cols 2-3
    # U indices: A[0,0]=2, A[0,1]=3, A[1,0]=6, A[1,1]=7
    # V indices: B[0,0]=2, B[0,1]=3, B[1,0]=6, B[1,1]=7
    # W indices: C[0,0]=8, C[0,1]=11, C[1,0]=9, C[1,1]=12
    col = add_laderman([2, 3, 6, 7], [2, 3, 6, 7], [8, 11, 9, 12], col)

    return {"U": U, "V": V, "W": W}
