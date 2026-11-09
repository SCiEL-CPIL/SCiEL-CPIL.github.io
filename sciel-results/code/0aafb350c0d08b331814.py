"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2(U, V, W, col_offset, a_row0, a_row1, b_col0, b_col1, c_row0, c_row1):
    """
    Add Strassen's 7-term decomposition for 2x2 * 2x2 matrix multiplication.
    
    A = [[a00, a01], [a10, a11]] (rows a_row0, a_row1; cols b_col0, b_col1)
    B = [[b00, b01], [b10, b11]] (rows a_row0, a_row1; cols b_col0, b_col1)
    C = [[c00, c01], [c10, c11]] (rows c_row0, c_row1; cols b_col0, b_col1)
    
    Strassen's 7 products:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b01)
    M7 = (a01-a11)*(b10+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    """
    col = col_offset
    # M1 = (a00+a11)*(b00+b11)
    # U: a00 + a11, V: b00 + b11
    U[a_row0 * 4 + b_col0][col] += 1
    U[a_row1 * 4 + b_col1][col] += 1
    V[b_col0 * 4 + a_row0][col] += 1  # b00 is at row a_row0 of B, col b_col0 -> index a_row0*4+b_col0
    V[b_col1 * 4 + a_row1][col] += 1  # b11 is at row a_row1 of B, col b_col1 -> index a_row1*4+b_col1
    
    # Wait, I need to be careful with indexing. Let me reconsider.
    # A is 2x4, indexed as A[i,j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x4, indexed as B[j,k] where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4, indexed as C[i,k] where i in {0,1}, k in {0,1,2,3}
    #
    # For the 2x2 block: A block uses rows a_row0,a_row1 of A and cols b_col0,b_col1 of A
    # B block uses rows b_col0,b_col1 of B and cols c_col0,c_col1 of B
    # C block uses rows c_row0,c_row1 of C and cols c_col0,c_col1 of C
    #
    # Let me redo this more carefully.
    pass

def _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
    """
    Add Strassen decomposition for computing C[c_r0:c_r1+1, c_c0:c_c1+1] 
    = A[a_r0:a_r1+1, b_c0:b_c1+1] * B[b_c0:b_c1+1, c_c0:c_c1+1]
    
    A is n x m, B is m x p, C is n x p
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    W index: k*n + i (trace convention W[k*n+i, r])
    
    a_r0, a_r1: rows of A (and C)
    b_c0, b_c1: cols of A (and rows of B)  
    c_c0, c_c1: cols of B (and C)
    """
    # Strassen's 7 products for 2x2 * 2x2
    # A_block = [[A[a_r0,b_c0], A[a_r0,b_c1]], [A[a_r1,b_c0], A[a_r1,b_c1]]]
    # B_block = [[B[b_c0,c_c0], B[b_c0,c_c1]], [B[b_c1,c_c0], B[b_c1,c_c1]]]
    # C_block = [[C[a_r0,c_c0], C[a_r0,c_c1]], [C[a_r1,c_c0], C[a_r1,c_c1]]]
    
    # M1 = (A[a_r0,b_c0] + A[a_r1,b_c1]) * (B[b_c0,c_c0] + B[b_c1,c_c1])
    U[a_r0 * 4 + b_c0][col] += 1
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c0 * 4 + c_c0][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] += M1, C[a_r1,c_c1] += M1
    W[c_c0 * 3 + a_r0][col] += 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M2 = (A[a_r1,b_c0] + A[a_r1,b_c1]) * B[b_c0,c_c0]
    U[a_r1 * 4 + b_c0][col] += 1
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c0 * 4 + c_c0][col] += 1
    # C[a_r1,c_c0] += M2, C[a_r1,c_c1] -= M2
    W[c_c0 * 3 + a_r1][col] += 1
    W[c_c1 * 3 + a_r1][col] -= 1
    col += 1
    
    # M3 = A[a_r0,b_c0] * (B[b_c0,c_c1] - B[b_c1,c_c1])
    U[a_r0 * 4 + b_c0][col] += 1
    V[b_c0 * 4 + c_c1][col] += 1
    V[b_c1 * 4 + c_c1][col] -= 1
    # C[a_r0,c_c1] += M3, C[a_r1,c_c1] += M3
    W[c_c1 * 3 + a_r0][col] += 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M4 = A[a_r1,b_c1] * (B[b_c1,c_c0] - B[b_c0,c_c0])
    U[a_r1 * 4 + b_c1][col] += 1
    V[b_c1 * 4 + c_c0][col] += 1
    V[b_c0 * 4 + c_c0][col] -= 1
    # C[a_r0,c_c0] += M4, C[a_r1,c_c0] += M4
    W[c_c0 * 3 + a_r0][col] += 1
    W[c_c0 * 3 + a_r1][col] += 1
    col += 1
    
    # M5 = (A[a_r0,b_c0] + A[a_r0,b_c1]) * B[b_c1,c_c1]
    U[a_r0 * 4 + b_c0][col] += 1
    U[a_r0 * 4 + b_c1][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] -= M5, C[a_r1,c_c1] += M5
    W[c_c0 * 3 + a_r0][col] -= 1
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M6 = (A[a_r1,b_c0] - A[a_r0,b_c0]) * (B[b_c0,c_c0] + B[b_c0,c_c1])
    U[a_r1 * 4 + b_c0][col] += 1
    U[a_r0 * 4 + b_c0][col] -= 1
    V[b_c0 * 4 + c_c0][col] += 1
    V[b_c0 * 4 + c_c1][col] += 1
    # C[a_r1,c_c1] += M6
    W[c_c1 * 3 + a_r1][col] += 1
    col += 1
    
    # M7 = (A[a_r0,b_c1] - A[a_r1,b_c1]) * (B[b_c1,c_c0] + B[b_c1,c_c1])
    U[a_r0 * 4 + b_c1][col] += 1
    U[a_r1 * 4 + b_c1][col] -= 1
    V[b_c1 * 4 + c_c0][col] += 1
    V[b_c1 * 4 + c_c1][col] += 1
    # C[a_r0,c_c0] += M7
    W[c_c0 * 3 + a_r0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Strategy: Split the 3x4 matrix into a 2x4 block and a 1x4 block.
    # 2x4 x 4x4: Use Strassen on four 2x2 sub-blocks -> rank 4*7 = 28
    # 1x4 x 4x4: Schoolbook -> rank 16
    # Total rank: 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (12, 44)
    V = [[0] * rank for _ in range(m * p)]  # (16, 44)
    W = [[0] * rank for _ in range(p * n)]  # (12, 44)
    
    col = 0
    
    # Part 1: 2x4 x 4x4 using Strassen (rows 0,1 of A)
    # Split into four 2x2 * 2x2 blocks:
    # Block (i,j) where i in {0,1} for row-pair, j in {0,1} for col-pair
    # A rows: {0,1}, A cols: {2j, 2j+1}
    # B rows: {2j, 2j+1}, B cols: {2i, 2i+1}
    # C rows: {0,1}, C cols: {2i, 2i+1}
    for j in range(2):  # A col pair / B row pair
        for i in range(2):  # B col pair / C col pair
            a_r0, a_r1 = 0, 1
            b_c0, b_c1 = 2*j, 2*j+1
            c_c0, c_c1 = 2*i, 2*i+1
            c_r0, c_r1 = 0, 1
            col = _add_strassen_2x2(U, V, W, col, a_r0, a_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1)
    
    # Part 2: 1x4 x 4x4 schoolbook (row 2 of A)
    for j in range(4):
        for k in range(4):
            U[2 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank} but used {col} columns"
    
    # Verify correctness
    # Build tensor from decomposition and compare to expected
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute product tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each output C[i,k] = sum_j A[i,j] * B[j,k]
    # The decomposition should give: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if (i,j,k) is valid
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    errors.append((i, j, k, val))
    
    if errors:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
