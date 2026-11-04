"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2*2x2 -> 2x2, rank 7.
    
    Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2*2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M4
    # c10 = M2 + M5
    # c11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Column 0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[3, 0] = 1  # c11
    
    # Column 1: M2 = (a10+a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = -1  # c11
    
    # Column 2: M3 = a00(b10-b11)
    U[0, 2] = 1  # a00
    V[2, 2] = 1  # b10
    V[3, 2] = -1  # b11
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # Column 3: M4 = a11(b01-b00)
    U[3, 3] = 1  # a11
    V[1, 3] = 1  # b01
    V[0, 3] = -1  # b00
    W[0, 3] = -1  # c00
    W[1, 3] = 1  # c01
    
    # Column 4: M5 = (a00+a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = 1  # c00
    W[2, 4] = 1  # c10
    
    # Column 5: M6 = (a10-a00)(b00+b10)
    U[2, 5] = 1  # a10
    U[0, 5] = -1  # a00
    V[0, 5] = 1  # b00
    V[2, 5] = 1  # b10
    W[3, 5] = 1  # c11
    
    # Column 6: M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1  # a11
    V[1, 6] = 1  # b01
    V[3, 6] = 1  # b11
    W[0, 6] = -1  # c00
    
    return U, V, W


def _schoolbook(n, m, p, i_off=0, j_off=0, k_off=0):
    """Schoolbook decomposition for n x m * m x p.
    
    Returns (U, V, W) with shapes (n*m, n*m*p), (m*p, n*m*p), (p*n, n*m*p).
    Indices are relative to offsets.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _combine_decomps(decomps, sizes):
    """Combine multiple decompositions into one.
    
    decomps: list of (U, V, W) tuples
    sizes: list of (n, m, p) for each subproblem
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    
    # Determine overall dimensions
    n_total = max(s[0] for s in sizes)
    m_total = max(s[1] for s in sizes)
    p_total = max(s[2] for s in sizes)
    
    U = np.zeros((n_total * m_total, total_rank))
    V = np.zeros((m_total * p_total, total_rank))
    W = np.zeros((p_total * n_total, total_rank))
    
    col_off = 0
    for (U_d, V_d, W_d), (n, m, p) in zip(decomps, sizes):
        r = U_d.shape[1]
        # Place U
        U[:n*m, col_off:col_off+r] = U_d
        # Place V
        V[:m*p, col_off:col_off+r] = V_d
        # Place W
        W[:p*n, col_off:col_off+r] = W_d
        col_off += r
    
    return U, V, W


def construct():
    """
    Decompose 4x5x6 matrix multiplication.
    
    Uses Strassen (rank 7) for 2x2x2 subproblems and schoolbook elsewhere.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split A (4x5) into:
    #   A_top (2x5), A_bot (2x5)
    # Split B (5x6) into:
    #   B_left (5x3), B_right (5x3)
    
    # C_top_left (2x3) = A_top * B_left
    # C_top_right (2x3) = A_top * B_right
    # C_bot_left (2x3) = A_bot * B_left
    # C_bot_right (2x3) = A_bot * B_right
    
    # Each is 2x5 * 5x3. Schoolbook rank = 2*5*3 = 30.
    # Total: 4 * 30 = 120. Same as before.
    
    # Let's try a different split.
    # Split inner dim 5 into 2+3:
    # A = [A1 (4x2) | A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 4x2 * 2x6, rank 4*2*6 = 48
    # A2*B2: 4x3 * 3x6, rank 4*3*6 = 72
    # Total: 120. Same.
    
    # To use Strassen, I need 2x2x2 subproblems.
    # Let me split more finely:
    # A (4x5): rows 1+1+1+1, cols 1+1+1+1+1
    # This gives 1x1*1x1 subproblems, each rank 1. Total 4*5*6=120.
    
    # The key is to group into 2x2 blocks where the middle dim is 2.
    
    # Let me try: 
    # A (4x5) -> split rows into 2+2, cols into 2+2+1
    # A = [[A11(2x2), A12(2x2), A13(2x1)],
    #      [A21(2x2), A22(2x2), A23(2x1)]]
    # B (5x6) -> split rows into 2+2+1, cols into 2+2+2
    # B = [[B11(2x2), B12(2x2), B13(2x2)],
    #      [B21(2x2), B22(2x2), B23(2x2)],
    #      [B31(1x2), B32(1x2), B33(1x2)]]
    
    # C (4x6) = A*B
    # C11(2x2) = A11*B11 + A12*B21 + A13*B31
    # C12(2x2) = A11*B12 + A12*B22 + A13*B32
    # C13(2x2) = A11*B13 + A12*B23 + A13*B33
    # C21(2x2) = A21*B11 + A22*B21 + A23*B31
    # C22(2x2) = A21*B12 + A22*B22 + A23*B32
    # C23(2x2) = A21*B13 + A22*B23 + A23*B33
    
    # Each Cij is 2x2, sum of three products:
    # - 2x2 * 2x2 (rank 7 with Strassen)
    # - 2x2 * 2x2 (rank 7 with Strassen)  
    # - 2x1 * 1x2 (rank 1)
    
    # So each Cij: rank 7 + 7 + 1 = 15
    # Total: 6 * 15 = 90
    
    # But wait, the factors are shared! A11 appears in C11, C12, C13.
    # B11 appears in C11, C21.
    # So I can't just sum the ranks independently.
    
    # This is the fundamental issue. Let me just use schoolbook.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
