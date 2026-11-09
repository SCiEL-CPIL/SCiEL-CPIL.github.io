import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen decomposition for 2x2x2
    # U_local: 2x7 (rows correspond to a00, a01, a10, a11? No, A is 2x2)
    # A = [[a00, a01], [a10, a11]]
    # U_local[i, c] is coeff for A[i, j] in multiplier c?
    # Actually U is n x rank. n=2.
    # U_local[0, :] corresponds to row 0 of A (a00, a01)
    # U_local[1, :] corresponds to row 1 of A (a10, a11)
    
    # Strassen Multipliers:
    # M1 = (a10+a11)b00
    # M2 = a11(b01-b00)
    # M3 = (a00-a10)b11
    # M4 = a00(b10+b11)
    # M5 = (a00+a01)b10
    # M6 = (a01-a11)(b00+b11)
    # M7 = (a10-a00)(b01+b11)
    
    # U_local (2x7)
    # Row 0 (a00, a01):
    # M1: 0
    # M2: 0
    # M3: 1 (a00)
    # M4: 1 (a00)
    # M5: 1 (a00)
    # M6: 0
    # M7: -1 (a00)
    U_local_row0 = [0, 0, 1, 1, 1, 0, -1]
    
    # Row 1 (a10, a11):
    # M1: 1 (a10), 1 (a11) -> But U is per row?
    # U is n x rank.
    # U[i, c] is the value in A[i, j] that goes into multiplier c?
    # No, U is the matrix such that A' = A U?
    # Standard: A'_{i,c} = sum_j A_{i,j} U_{j,c}.
    # So U is m x rank? No, U is n x rank?
    # Task says: U (n*m, rank).
    # My local A is 2x2. n=2, m=2.
    # U_local should be 4x7?
    # Let's check the interface.
    # U (n*m, rank).
    # For 2x2x2, U is 4x7.
    # Rows correspond to A indices: (0,0), (0,1), (1,0), (1,1).
    # U[i*m+j, c].
    
    U_local = [[0]*7 for _ in range(4)]
    # M1: a10, a11
    U_local[2][0] = 1 # a10
    U_local[3][0] = 1 # a11
    # M2: a11
    U_local[3][1] = 1
    # M3: a00, a10
    U_local[0][2] = 1
    U_local[2][2] = -1
    # M4: a00
    U_local[0][3] = 1
    # M5: a00, a01
    U_local[0][4] = 1
    U_local[1][4] = 1
    # M6: a01, a11
    U_local[1][5] = 1
    U_local[3][5] = -1
    # M7: a10, a00
    U_local[2][6] = 1
    U_local[0][6] = -1

    V_local = [[0]*7 for _ in range(4)]
    # M1: b00
    V_local[0][0] = 1
    # M2: b01, b00
    V_local[1][1] = 1
    V_local[0][1] = -1
    # M3: b11
    V_local[3][2] = 1
    # M4: b10, b11
    V_local[2][3] = 1
    V_local[3][3] = 1
    # M5: b10
    V_local[2][4] = 1
    # M6: b00, b11
    V_local[0][5] = 1
    V_local[3][5] = 1
    # M7: b01, b11
    V_local[1][6] = 1
    V_local[3][6] = 1

    W_local = [[0]*7 for _ in range(4)]
    # C00 = M1 - M2 + M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    W_local[0][0] = 1 # M1
    W_local[0][1] = -1 # M2
    W_local[0][3] = 1 # M4
    W_local[0][5] = 1 # M6
    
    W_local[1][4] = 1 # M5
    W_local[1][6] = 1 # M7
    
    W_local[2][1] = 1 # M2
    W_local[2][3] = 1 # M4
    
    W_local[3][0] = 1 # M1
    W_local[3][1] = -1 # M2
    W_local[3][2] = 1 # M3
    W_local[3][4] = 1 # M5

    # Map to Global
    # Sub-prob 1: A1, B11. Cols 0-6
    # A1 is A cols 0,1.
    for c in range(7):
        for r in range(4):
            U[0*2+r][c] = U_local[r][c]
        for r in range(4):
            V[0*4+r][c] = V_local[r][c]
        for r in range(4):
            W[0*2+r][c] = W_local[r][c]

    # Sub-prob 2: A2, B31. Cols 7-13
    # A2 is A cols 2,3.
    # B31 is B rows 2,3, cols 0,1.
    for c in range(7):
        col = 7 + c
        # U: A2 rows 2,3 (global index 2,3)
        U[2][col] = U_local[0][c] # a00 of A2 is A[0,2] -> global U[0*2+2]? No.
        # A is 2x4. Index i*m+j.
        # A2 is cols 2,3.
        # A[0,2] is index 2. A[0,3] is index 3.
        # A[1,2] is index 6. A[1,3] is index 7.
        # Local A2: (0,0)->A[0,2], (0,1)->A[0,3], (1,0)->A[1,2], (1,1)->A[1,3]
        U[2][col] = U_local[0][c]
        U[3][col] = U_local[1][c]
        U[6][col] = U_local[2][c]
        U[7][col] = U_local[3][c]
        
        # V: B31 rows 2,3, cols 0,1.
        # B[2,0] is index 2*4+0 = 8.
        # B[2,1] is index 9.
        # B[3,0] is index 12.
        # B[3,1] is index 13.
        V[8][col] = V_local[0][c]
        V[9][col] = V_local[1][c]
        V[12][col] = V_local[2][c]
        V[13][col] = V_local[3][c]
        
        # W: Same as Sub-prob 1 (adds to C[:, 0:2])
        W[0][col] = W_local[0][c]
        W[1][col] = W_local[1][c]
        W[2][col] = W_local[2][c]
        W[3][col] = W_local[3][c]

    # Sub-prob 3: A1, B12. Cols 14-20
    # B12 is B rows 0,1, cols 2,3.
    for c in range(7):
        col = 14 + c
        U[0][col] = U_local[0][c]
        U[1][col] = U_local[1][c]
        U[4][col] = U_local[2][c]
        U[5][col] = U_local[3][c]
        
        # B12: B[0,2] idx 2, B[0,3] idx 3, B[1,2] idx 6, B[1,3] idx 7
        V[2][col] = V_local[0][c]
        V[3][col] = V_local[1][c]
        V[6][col] = V_local[2][c]
        V[7][col] = V_local[3][c]
        
        # W: Adds to C[:, 2:4]
        W[4][col] = W_local[0][c]
        W[5][col] = W_local[1][c]
        W[6][col] = W_local[2][c]
        W[7][col] = W_local[3][c]

    # Sub-prob 4: A2, B32. Cols 21-27
    # B32 is B rows 2,3, cols 2,3.
    for c in range(7):
        col = 21 + c
        U[2][col] = U_local[0][c]
        U[3][col] = U_local[1][c]
        U[6][col] = U_local[2][c]
        U[7][col] = U_local[3][c]
        
        # B32: B[2,2] idx 10, B[2,3] idx 11, B[3,2] idx 14, B[3,3] idx 15
        V[10][col] = V_local[0][c]
        V[11][col] = V_local[1][c]
        V[14][col] = V_local[2][c]
        V[15][col] = V_local[3][c]
        
        # W: Adds to C[:, 2:4]
        W[4][col] += W_local[0][c]
        W[5][col] += W_local[1][c]
        W[6][col] += W_local[2][c]
        W[7][col] += W_local[3][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
