import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    Uses Strassen-based decomposition (rank 42).
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Strassen 2x2 decomposition:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    # Process A0 (A columns 0-1) times each 2x2 block of B_top (B rows 0-1, columns 2t-2t+1)
    # And A1 (A columns 2-3) times each 2x2 block of B_bottom (B rows 2-3, columns 2t-2t+1)
    
    # A is 2x4. Split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x6. Split rows into B_top (rows 0,1) and B_bot (rows 2,3).
    # Split B cols into 3 blocks of 2: cols 0,1; 2,3; 4,5.
    
    # For each column block t (0, 1, 2):
    #   C[:, 2t:2t+2] = A_left * B_top[:, 2t:2t+2] + A_right * B_bot[:, 2t:2t+2]
    
    for t in range(3):
        b_col0 = 2 * t
        b_col1 = 2 * t + 1
        
        # Part 1: A_left (cols 0,1) * B_top (rows 0,1)
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j' in {0,1}, k in {2t, 2t+1}
        # C indices: i' in {0,1}, k' in {2t, 2t+1}
        
        # Strassen terms for A_left * B_top
        # Local A: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # Local B: b00=B[0,2t], b01=B[0,2t+1], b10=B[1,2t], b11=B[1,2t+1]
        
        # M1 = (a00+a11)*(b00+b11)
        # C00 += M1, C11 += M1
        U[0 * m + 0][col] += 1 # A[0,0]
        U[1 * m + 1][col] += 1 # A[1,1]
        V[0 * p + b_col0][col] += 1 # B[0,2t]
        V[1 * p + b_col1][col] += 1 # B[1,2t+1]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M2 = (a10+a11)*b00
        # C10 += M2, C11 -= M2
        U[1 * m + 0][col] += 1 # A[1,0]
        U[1 * m + 1][col] += 1 # A[1,1]
        V[0 * p + b_col0][col] += 1 # B[0,2t]
        W[b_col0 * n + 1][col] += 1 # C[1,2t]
        W[b_col1 * n + 1][col] += -1 # C[1,2t+1]
        col += 1

        # M3 = a00*(b01-b11)
        # C01 += M3, C11 += M3
        U[0 * m + 0][col] += 1 # A[0,0]
        V[0 * p + b_col1][col] += 1 # B[0,2t+1]
        V[1 * p + b_col1][col] += -1 # B[1,2t+1]
        W[b_col1 * n + 0][col] += 1 # C[0,2t+1]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M4 = a11*(b10-b00)
        # C00 += M4, C10 += M4
        U[1 * m + 1][col] += 1 # A[1,1]
        V[1 * p + b_col0][col] += 1 # B[1,2t]
        V[0 * p + b_col0][col] += -1 # B[0,2t]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        W[b_col0 * n + 1][col] += 1 # C[1,2t]
        col += 1

        # M5 = (a00+a01)*b11
        # C00 -= M5, C01 += M5
        U[0 * m + 0][col] += 1 # A[0,0]
        U[0 * m + 1][col] += 1 # A[0,1]
        V[1 * p + b_col1][col] += 1 # B[1,2t+1]
        W[b_col0 * n + 0][col] += -1 # C[0,2t]
        W[b_col1 * n + 0][col] += 1 # C[0,2t+1]
        col += 1

        # M6 = (a10-a00)*(b00+b01)
        # C11 += M6
        U[1 * m + 0][col] += 1 # A[1,0]
        U[0 * m + 0][col] += -1 # A[0,0]
        V[0 * p + b_col0][col] += 1 # B[0,2t]
        V[0 * p + b_col1][col] += 1 # B[0,2t+1]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M7 = (a01-a11)*(b10+b11)
        # C00 += M7
        U[0 * m + 1][col] += 1 # A[0,1]
        U[1 * m + 1][col] += -1 # A[1,1]
        V[1 * p + b_col0][col] += 1 # B[1,2t]
        V[1 * p + b_col1][col] += 1 # B[1,2t+1]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        col += 1

        # Part 2: A_right (cols 2,3) * B_bot (rows 2,3)
        # A indices: i in {0,1}, j in {2,3}
        # B indices: j' in {2,3}, k in {2t, 2t+1}
        # C indices: i' in {0,1}, k' in {2t, 2t+1}
        
        # Local A: a00=A[0,2], a01=A[0,3], a10=A[1,2], a11=A[1,3]
        # Local B: b00=B[2,2t], b01=B[2,2t+1], b10=B[3,2t], b11=B[3,2t+1]
        
        # M1 = (a00+a11)*(b00+b11)
        U[0 * m + 2][col] += 1 # A[0,2]
        U[1 * m + 3][col] += 1 # A[1,3]
        V[2 * p + b_col0][col] += 1 # B[2,2t]
        V[3 * p + b_col1][col] += 1 # B[3,2t+1]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M2 = (a10+a11)*b00
        U[1 * m + 2][col] += 1 # A[1,2]
        U[1 * m + 3][col] += 1 # A[1,3]
        V[2 * p + b_col0][col] += 1 # B[2,2t]
        W[b_col0 * n + 1][col] += 1 # C[1,2t]
        W[b_col1 * n + 1][col] += -1 # C[1,2t+1]
        col += 1

        # M3 = a00*(b01-b11)
        U[0 * m + 2][col] += 1 # A[0,2]
        V[2 * p + b_col1][col] += 1 # B[2,2t+1]
        V[3 * p + b_col1][col] += -1 # B[3,2t+1]
        W[b_col1 * n + 0][col] += 1 # C[0,2t+1]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M4 = a11*(b10-b00)
        U[1 * m + 3][col] += 1 # A[1,3]
        V[3 * p + b_col0][col] += 1 # B[3,2t]
        V[2 * p + b_col0][col] += -1 # B[2,2t]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        W[b_col0 * n + 1][col] += 1 # C[1,2t]
        col += 1

        # M5 = (a00+a01)*b11
        U[0 * m + 2][col] += 1 # A[0,2]
        U[0 * m + 3][col] += 1 # A[0,3]
        V[3 * p + b_col1][col] += 1 # B[3,2t+1]
        W[b_col0 * n + 0][col] += -1 # C[0,2t]
        W[b_col1 * n + 0][col] += 1 # C[0,2t+1]
        col += 1

        # M6 = (a10-a00)*(b00+b01)
        U[1 * m + 2][col] += 1 # A[1,2]
        U[0 * m + 2][col] += -1 # A[0,2]
        V[2 * p + b_col0][col] += 1 # B[2,2t]
        V[2 * p + b_col1][col] += 1 # B[2,2t+1]
        W[b_col1 * n + 1][col] += 1 # C[1,2t+1]
        col += 1

        # M7 = (a01-a11)*(b10+b11)
        U[0 * m + 3][col] += 1 # A[0,3]
        U[1 * m + 3][col] += -1 # A[1,3]
        V[3 * p + b_col0][col] += 1 # B[3,2t]
        V[3 * p + b_col1][col] += 1 # B[3,2t+1]
        W[b_col0 * n + 0][col] += 1 # C[0,2t]
        col += 1

    return {"U": U, "V": V, "W": W}
