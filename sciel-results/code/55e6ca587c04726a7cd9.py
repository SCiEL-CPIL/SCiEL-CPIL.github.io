"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _strassen_block(A_idx, B_idx, C_idx, U, V, W, start_col):
    """
    Adds a Strassen decomposition for a 2x2x2 block.
    A_idx: list of 4 indices in U (row, col) -> (i*m + j)
    B_idx: list of 4 indices in V (row, col) -> (j*p + k)
    C_idx: list of 4 indices in W (row, col) -> (k*n + i)
    Returns next column index.
    """
    # A_idx = [A11, A12, A21, A22]
    # B_idx = [B11, B12, B21, B22]
    # C_idx = [C11, C12, C21, C22]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[A_idx[0]][start_col] += 1
    U[A_idx[3]][start_col] += 1
    V[B_idx[0]][start_col] += 1
    V[B_idx[3]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    W[C_idx[3]][start_col] += 1
    start_col += 1
    
    # M2 = (A21 + A22)B11
    U[A_idx[2]][start_col] += 1
    U[A_idx[3]][start_col] += 1
    V[B_idx[0]][start_col] += 1
    W[C_idx[2]][start_col] += 1
    W[C_idx[3]][start_col] += -1
    start_col += 1
    
    # M3 = A11(B12 - B22)
    U[A_idx[0]][start_col] += 1
    V[B_idx[1]][start_col] += 1
    V[B_idx[3]][start_col] += -1
    W[C_idx[1]][start_col] += 1
    W[C_idx[0]][start_col] += -1
    start_col += 1
    
    # M4 = A22(B21 - B11)
    U[A_idx[3]][start_col] += 1
    V[B_idx[2]][start_col] += 1
    V[B_idx[0]][start_col] += -1
    W[C_idx[2]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    start_col += 1
    
    # M5 = (A11 + A12)B22
    U[A_idx[0]][start_col] += 1
    U[A_idx[1]][start_col] += 1
    V[B_idx[3]][start_col] += 1
    W[C_idx[1]][start_col] += 1
    W[C_idx[0]][start_col] += -1
    start_col += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[A_idx[2]][start_col] += 1
    U[A_idx[0]][start_col] += -1
    V[B_idx[0]][start_col] += 1
    V[B_idx[1]][start_col] += 1
    W[C_idx[3]][start_col] += 1
    start_col += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[A_idx[1]][start_col] += 1
    U[A_idx[3]][start_col] += -1
    V[B_idx[2]][start_col] += 1
    V[B_idx[3]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    W[C_idx[3]][start_col] += 1
    start_col += 1
    
    return start_col

def _tail_block(A_idx, B_idx, C_idx, U, V, W, start_col):
    """
    Adds a schoolbook decomposition for a 2x4x1 block.
    A_idx: list of 4 indices in U (row 1 and 2, cols 1-4) -> actually 8 indices?
    No, A_idx should be the 8 A elements?
    Let's pass specific indices.
    A_idx: [A11, A12, A13, A14, A21, A22, A23, A24]
    B_idx: [B15, B25, B35, B45]
    C_idx: [C15, C25]
    """
    # P1 = A11 B15
    U[A_idx[0]][start_col] += 1
    V[B_idx[0]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    start_col += 1
    
    # P2 = A12 B25
    U[A_idx[1]][start_col] += 1
    V[B_idx[1]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    start_col += 1
    
    # P3 = A13 B35
    U[A_idx[2]][start_col] += 1
    V[B_idx[2]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    start_col += 1
    
    # P4 = A14 B45
    U[A_idx[3]][start_col] += 1
    V[B_idx[3]][start_col] += 1
    W[C_idx[0]][start_col] += 1
    start_col += 1
    
    # P5 = A21 B15
    U[A_idx[4]][start_col] += 1
    V[B_idx[0]][start_col] += 1
    W[C_idx[1]][start_col] += 1
    start_col += 1
    
    # P6 = A22 B25
    U[A_idx[5]][start_col] += 1
    V[B_idx[1]][start_col] += 1
    W[C_idx[1]][start_col] += 1
    start_col += 1
    
    # P7 = A23 B35
    U[A_idx[6]][start_col] += 1
    V[B_idx[2]][start_col] += 1
    W[C_idx[1]][start_col] += 1
    start_col += 1
    
    # P8 = A24 B45
    U[A_idx[7]][start_col] += 1
    V[B_idx[3]][start_col] += 1
    W[C_idx[1]][start_col] += 1
    start_col += 1
    
    return start_col

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 36
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Indices for U (A matrix 2x4)
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    A11, A12, A13, A14 = 0, 1, 2, 3
    A21, A22, A23, A24 = 4, 5, 6, 7
    
    # Indices for V (B matrix 4x5)
    # Row 0: 0, 1, 2, 3, 4
    # Row 1: 5, 6, 7, 8, 9
    # Row 2: 10, 11, 12, 13, 14
    # Row 3: 15, 16, 17, 18, 19
    B11, B12, B13, B14, B15 = 0, 1, 2, 3, 4
    B21, B22, B23, B24, B25 = 5, 6, 7, 8, 9
    B31, B32, B33, B34, B35 = 10, 11, 12, 13, 14
    B41, B42, B43, B44, B45 = 15, 16, 17, 18, 19
    
    # Indices for W (C matrix 2x5)
    # Row 0: 0, 1, 2, 3, 4
    # Row 1: 5, 6, 7, 8, 9
    C11, C12, C13, C14, C15 = 0, 1, 2, 3, 4
    C21, C22, C23, C24, C25 = 5, 6, 7, 8, 9
    
    col = 0
    
    # Block 1: A11, A12 x B11, B12, B21, B22 -> C11, C12, C21, C22?
    # Wait, my Strassen block assumes 2x2 A and 2x2 B.
    # A part: Row 1, Cols 1-2. A_idx = [A11, A12, A21, A22]?
    # No, Block 1 should be A[0, 0:2] and B[0:2, 0:2].
    # A[0, 0:2] is A11, A12. But Strassen needs 2x2 A.
    # My A is 2x4.
    # I split A into A_L (2x2) and A_R (2x2).
    # A_L = [[A11, A12], [A21, A22]].
    # A_R = [[A13, A14], [A23, A24]].
    
    # B is 4x4.
    # B_LL = [[B11, B12], [B21, B22]].
    # B_LR = [[B13, B14], [B23, B24]].
    # B_RL = [[B31, B32], [B41, B42]].
    # B_RR = [[B33, B34], [B43, B44]].
    
    # C is 2x4.
    # C_L = [[C11, C12], [C21, C22]].
    # C_R = [[C13, C14], [C23, C24]].
    
    # Product 1: A_L * B_LL -> C_L
    # A_idx = [A11, A12, A21, A22]
    # B_idx = [B11, B12, B21, B22]
    # C_idx = [C11, C12, C21, C22]
    col = _strassen_block([A11, A12, A21, A22], [B11, B12, B21, B22], [C11, C12, C21, C22], U, V, W, col)
    
    # Product 2: A_L * B_LR -> C_R?
    # No, A_L * B_LR gives C[0, 2:4] and C[1, 2:4]?
    # A_L is rows 1-2. B_LR is rows 1-2, cols 3-4.
    # Result is rows 1-2, cols 3-4.
    # C_idx = [C13, C14, C23, C24]
    # B_idx = [B13, B14, B23, B24]
    col = _strassen_block([A11, A12, A21, A22], [B13, B14, B23, B24], [C13, C14, C23, C24], U, V, W, col)
    
    # Product 3: A_R * B_RL -> C_L?
    # A_R is rows 1-2, cols 3-4. B_RL is rows 3-4, cols 1-2.
    # Result is rows 1-2, cols 1-2.
    # C_idx = [C11, C12, C21, C22]
    # B_idx = [B31, B32, B41, B42]
    # A_idx = [A13, A14, A23, A24]
    col = _strassen_block([A13, A14, A23, A24], [B31, B32, B41, B42], [C11, C12, C21, C22], U, V, W, col)
    
    # Product 4: A_R * B_RR -> C_R?
    # A_R is rows 1-2, cols 3-4. B_RR is rows 3-4, cols 3-4.
    # Result is rows 1-2, cols 3-4.
    # C_idx = [C13, C14, C23, C24]
    # B_idx = [B33, B34, B43, B44]
    # A_idx = [A13, A14, A23, A24]
    col = _strassen_block([A13, A14, A23, A24], [B33, B34, B43, B44], [C13, C14, C23, C24], U, V, W, col)
    
    # Tail: A * B_col5 -> C_col5
    # A_idx = [A11, A12, A13, A14, A21, A22, A23, A24]
    # B_idx = [B15, B25, B35, B45]
    # C_idx = [C15, C25]
    col = _tail_block([A11, A12, A13, A14, A21, A22, A23, A24], [B15, B25, B35, B45], [C15, C25], U, V, W, col)
    
    if col != rank:
        raise ValueError(f"Column mismatch: {col} != {rank}")
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
