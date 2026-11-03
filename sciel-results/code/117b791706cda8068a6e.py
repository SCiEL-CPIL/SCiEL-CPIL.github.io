"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Strassen-based decomposition for 2x4x6
    # We decompose into 6 blocks of 2x2x2 Strassen multiplications
    # Each Strassen 2x2x2 has rank 7
    # Total rank: 6 * 7 = 42

    rank = 42

    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen 2x2x2 decomposition coefficients
    # For multiplying 2x2 matrix A by 2x2 matrix B to get 2x2 result C:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def add_strassen(a_rows, b_cols, w_rows, start):
        """
        Add a Strassen 2x2x2 block.
        a_rows: indices [a00, a01, a10, a11] in U (row indices)
        b_cols: indices [b00, b01, b10, b11] in V (row indices)
        w_rows: indices [c00, c01, c10, c11] in W (row indices)
        start: starting column index
        """
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_cols
        c00, c01, c10, c11 = w_rows

        # M1 = (a00+a11)(b00+b11)
        U[a00][start+0] += 1
        U[a11][start+0] += 1
        V[b00][start+0] += 1
        V[b11][start+0] += 1
        W[c00][start+0] += 1
        W[c11][start+0] += 1

        # M2 = (a10+a11)b00
        U[a10][start+1] += 1
        U[a11][start+1] += 1
        V[b00][start+1] += 1
        W[c10][start+1] += 1
        W[c11][start+1] -= 1

        # M3 = a00(b10-b11)
        U[a00][start+2] += 1
        V[b10][start+2] += 1
        V[b11][start+2] -= 1
        W[c01][start+2] += 1
        W[c11][start+2] += 1

        # M4 = a11(b01-b00)
        U[a11][start+3] += 1
        V[b01][start+3] += 1
        V[b00][start+3] -= 1
        W[c00][start+3] += 1
        W[c10][start+3] += 1

        # M5 = (a00+a01)b11
        U[a00][start+4] += 1
        U[a01][start+4] += 1
        V[b11][start+4] += 1
        W[c00][start+4] -= 1
        W[c01][start+4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[a10][start+5] += 1
        U[a00][start+5] -= 1
        V[b00][start+5] += 1
        V[b01][start+5] += 1
        W[c11][start+5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[a01][start+6] += 1
        U[a11][start+6] -= 1
        V[b10][start+6] += 1
        V[b11][start+6] += 1
        W[c00][start+6] += 1
        W[c01][start+6] -= 1

    # Matrix A is 2x4, indexed as A[i,j] with i=0,1 and j=0,1,2,3
    # U row index: i*m + j = i*4 + j
    # So A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    #    A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7

    # Matrix B is 4x6, indexed as B[j,k] with j=0,1,2,3 and k=0,1,2,3,4,5
    # V row index: j*p + k = j*6 + k
    # So B[0,0]=0, B[0,1]=1, ..., B[0,5]=5
    #    B[1,0]=6, ..., B[1,5]=11
    #    B[2,0]=12, ..., B[2,5]=17
    #    B[3,0]=18, ..., B[3,5]=23

    # Result C is 2x6, indexed as C[i,k] with i=0,1 and k=0,1,2,3,4,5
    # W row index: k*n + i = k*2 + i
    # So C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, ..., C[0,5]=10, C[1,5]=11

    # We need to compute C = A * B
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

    # Split into 6 blocks based on j pairs:
    # Block 1: j=0,1 (A rows 0-1 for cols 0,1; B rows 0,1)
    # Block 2: j=2,3 (A rows 0-1 for cols 2,3; B rows 2,3)
    # But we have 4 columns in A, so we need to handle them differently.
    
    # Actually, let's think of it as:
    # C[:, 0:6] = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    
    # A[:, 0:2] is 2x2: [[A00, A01], [A10, A11]] -> U rows [0,1,4,5]
    # A[:, 2:4] is 2x2: [[A02, A03], [A12, A13]] -> U rows [2,3,6,7]
    
    # B[0:2, :] is 2x6, B[2:4, :] is 2x6
    
    # For each 2x2 block of A, we multiply by each 2x6 block of B
    # B has 3 such 2x6 blocks: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    # and similarly for B[2:4, :]
    
    # So total: 2 (A blocks) * 3 (B blocks) = 6 Strassen multiplications
    # Each produces a 2x2 result, contributing to different columns of C

    # Block (a_block, b_block):
    # a_block=0: A rows [0,1,4,5] (cols 0,1 of A)
    # a_block=1: A rows [2,3,6,7] (cols 2,3 of A)
    # b_block=0: B rows for cols 0,1 -> V rows [0,1,6,7]
    # b_block=1: B rows for cols 2,3 -> V rows [2,3,8,9]
    # b_block=2: B rows for cols 4,5 -> V rows [4,5,10,11]
    # w_block: result rows for corresponding C columns
    # b_block=0: C cols 0,1 -> W rows [0,2,1,3] (C[0,0],C[0,1],C[1,0],C[1,1])
    # b_block=1: C cols 2,3 -> W rows [4,6,5,7]
    # b_block=2: C cols 4,5 -> W rows [8,10,9,11]

    a_block_0 = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
    a_block_1 = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]

    b_block_0 = [0, 1, 6, 7]    # B[0,0], B[0,1], B[1,0], B[1,1]
    b_block_1 = [2, 3, 8, 9]    # B[0,2], B[0,3], B[1,2], B[1,3]
    b_block_2 = [4, 5, 10, 11]  # B[0,4], B[0,5], B[1,4], B[1,5]

    w_block_0 = [0, 2, 1, 3]    # C[0,0], C[0,1], C[1,0], C[1,1]
    w_block_1 = [4, 6, 5, 7]    # C[0,2], C[0,3], C[1,2], C[1,3]
    w_block_2 = [8, 10, 9, 11]  # C[0,4], C[0,5], C[1,4], C[1,5]

    # 6 Strassen blocks
    add_strassen(a_block_0, b_block_0, w_block_0, 0)
    add_strassen(a_block_0, b_block_1, w_block_1, 7)
    add_strassen(a_block_0, b_block_2, w_block_2, 14)
    add_strassen(a_block_1, b_block_0, w_block_0, 21)
    add_strassen(a_block_1, b_block_1, w_block_1, 28)
    add_strassen(a_block_1, b_block_2, w_block_2, 35)

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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

    # Try to reduce rank by removing redundant factors
    def try_remove_factor(U_list, V_list, W_list, r_to_remove, n, m, p):
        rank = len(U_list[0])
        new_rank = rank - 1
        U_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in U_list]
        V_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in V_list]
        W_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in W_list]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(new_rank):
                        total += U_new[i * m + j][r] * V_new[j * p + k][r] * W_new[k * n + i][r]
                    if total != 1:
                        return None
        return {"U": U_new, "V": V_new, "W": W_new}

    # Try removing each factor one at a time
    for r in range(rank):
        if time.time() > deadline:
            break
        result = try_remove_factor(U, V, W, r, n, m, p)
        if result is not None:
            return result

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
