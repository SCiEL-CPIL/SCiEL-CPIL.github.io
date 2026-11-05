import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use 2x (3x3x4 rank 32) decomposition
    # 3x3x4 rank 32 = 3x3x3 Winograd (23) + 3x3x1 Schoolbook (9)
    
    # Winograd 3x3x3 matrices
    # These are hardcoded from a known decomposition
    U_win = np.zeros((9, 23))
    V_win = np.zeros((9, 23))
    W_win = np.zeros((9, 23))
    
    # Placeholder for Winograd matrices
    # This is a risk, but I will try to use a known set
    # If this fails, I will fall back to schoolbook
    
    # Schoolbook for 3x3x1
    U_school = np.zeros((9, 9))
    V_school = np.zeros((9, 9))
    W_school = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            col = i * 3 + j
            U_school[i * 3 + j, col] = 1
            V_school[i * 3 + j, col] = 1
            W_school[i * 3 + j, col] = 1
            
    # Combine
    U = np.zeros((9, 32))
    V = np.zeros((9, 32))
    W = np.zeros((9, 32))
    
    # First 23 columns: Winograd
    U[:, :23] = U_win
    V[:, :23] = V_win
    W[:, :23] = W_win
    
    # Last 9 columns: Schoolbook
    U[:, 23:] = U_school
    V[:, 23:] = V_school
    W[:, 23:] = W_school
    
    # Now extend to 3x3x8
    # U is 9x32 for 3x3x4
    # V is 9x32 for 3x3x4 (but B is 3x4)
    # W is 9x32 for 3x3x4 (but C is 3x4)
    
    # For 3x3x8, we have two 3x3x4 blocks
    U_8 = np.zeros((9, 64))
    V_8 = np.zeros((24, 64)) # B is 3x8
    W_8 = np.zeros((24, 64)) # C is 3x8
    
    # Block 1: columns 0-3 of B
    U_8[:, :32] = U
    V_8[:9, :32] = V # B rows 0-2
    W_8[:9, :32] = W # C rows 0-2
    
    # Block 2: columns 4-7 of B
    U_8[:, 32:64] = U
    V_8[9:18, 32:64] = V # B rows 3-5 (but B is 3x8, so rows 0-2 for each block? No)
    
    # Wait, B is 3x8.
    # Block 1: B[:, 0:4]
    # Block 2: B[:, 4:8]
    # So V_8 rows 0-2 are for B[:, 0:4] and B[:, 4:8]?
    # No, V_8 has 24 rows.
    # Rows 0-2: B[:, 0]
    # Rows 3-5: B[:, 1]
    # ...
    # Rows 21-23: B[:, 7]
    
    # So for Block 1 (cols 0-3), V_8 rows 0-8 are used?
    # No, V is 9x32.
    # V[i, j] corresponds to A[i, j] and B[j, k].
    # For 3x3x4, B is 3x4.
    # V has 9 rows (3x3).
    # But B is 3x4, so V should have 12 rows?
    # No, V is (m*p, rank) = (3*4, 32) = (12, 32).
    
    # My previous U, V, W were for 3x3x3.
    # U (9x23), V (9x23), W (9x23).
    # For 3x3x4, V should be (12, 32).
    
    # This is getting complicated.
    # I will fall back to schoolbook 72.
    
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
# EVOLVE-BLOCK-END
