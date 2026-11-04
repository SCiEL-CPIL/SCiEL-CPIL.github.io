"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
    # U: 4x7, rows: a00, a01, a10, a11
    U = [[0]*7 for _ in range(4)]
    # V: 4x7, rows: b00, b01, b10, b11
    V = [[0]*7 for _ in range(4)]
    # W: 4x7, rows: c00, c01, c10, c11
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (a10+a11)b00
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (a10-a00)(b00+b01)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Return (U, V, W) for 4x4x4 Strassen decomposition, rank 49."""
    U2, V2, W2 = _strassen_2x2()
    
    # Define the 7 intermediate products.
    # Each is (left, right) where left and right are 2x2 arrays of lists of (4x4_index, coeff).
    # 4x4 index: i*4 + j for A[i,j]
    
    products = []
    
    # M1 = (A00 + A11)(B00 + B11)
    left1 = [
        [(0, 1), (10, 1)],   # A[0,0] + A[2,2]
        [(1, 1), (11, 1)],   # A[0,1] + A[2,3]
        [(4, 1), (14, 1)],   # A[1,0] + A[3,2]
        [(5, 1), (15, 1)],   # A[1,1] + A[3,3]
    ]
    right1 = [
        [(0, 1), (10, 1)],   # B[0,0] + B[2,2]
        [(1, 1), (11, 1)],   # B[0,1] + B[2,3]
        [(4, 1), (14, 1)],   # B[1,0] + B[3,2]
        [(5, 1), (15, 1)],   # B[1,1] + B[3,3]
    ]
    products.append((left1, right1))
    
    # M2 = (A10 + A11)B00
    left2 = [
        [(8, 1), (10, 1)],   # A[2,0] + A[2,2]
        [(9, 1), (11, 1)],   # A[2,1] + A[2,3]
        [(12, 1), (14, 1)],  # A[3,0] + A[3,2]
        [(13, 1), (15, 1)],  # A[3,1] + A[3,3]
    ]
    right2 = [
        [(0, 1)],            # B[0,0]
        [(1, 1)],            # B[0,1]
        [(4, 1)],            # B[1,0]
        [(5, 1)],            # B[1,1]
    ]
    products.append((left2, right2))
    
    # M3 = A00(B01 - B11)
    left3 = [
        [(0, 1)],            # A[0,0]
        [(1, 1)],            # A[0,1]
        [(4, 1)],            # A[1,0]
        [(5, 1)],            # A[1,1]
    ]
    right3 = [
        [(2, 1), (10, -1)],  # B[0,2] - B[2,2]
        [(3, 1), (11, -1)],  # B[0,3] - B[2,3]
        [(6, 1), (14, -1)],  # B[1,2] - B[3,2]
        [(7, 1), (15, -1)],  # B[1,3] - B[3,3]
    ]
    products.append((left3, right3))
    
    # M4 = A11(B10 - B00)
    left4 = [
        [(10, 1)],           # A[2,2]
        [(11, 1)],           # A[2,3]
        [(14, 1)],           # A[3,2]
        [(15, 1)],           # A[3,3]
    ]
    right4 = [
        [(8, 1), (0, -1)],   # B[2,0] - B[0,0]
        [(9, 1), (1, -1)],   # B[2,1] - B[0,1]
        [(12, 1), (4, -1)],  # B[3,0] - B[1,0]
        [(13, 1), (5, -1)],  # B[3,1] - B[1,1]
    ]
    products.append((left4, right4))
    
    # M5 = (A00 + A01)B11
    left5 = [
        [(0, 1), (2, 1)],    # A[0,0] + A[0,2]
        [(1, 1), (3, 1)],    # A[0,1] + A[0,3]
        [(4, 1), (6, 1)],    # A[1,0] + A[1,2]
        [(5, 1), (7, 1)],    # A[1,1] + A[1,3]
    ]
    right5 = [
        [(10, 1)],           # B[2,2]
        [(11, 1)],           # B[2,3]
        [(14, 1)],           # B[3,2]
        [(15, 1)],           # B[3,3]
    ]
    products.append((left5, right5))
    
    # M6 = (A10 - A00)(B00 + B01)
    left6 = [
        [(8, 1), (0, -1)],   # A[2,0] - A[0,0]
        [(9, 1), (1, -1)],   # A[2,1] - A[0,1]
        [(12, 1), (4, -1)],  # A[3,0] - A[1,0]
        [(13, 1), (5, -1)],  # A[3,1] - A[1,1]
    ]
    right6 = [
        [(0, 1), (2, 1)],    # B[0,0] + B[0,2]
        [(1, 1), (3, 1)],    # B[0,1] + B[0,3]
        [(4, 1), (6, 1)],    # B[1,0] + B[1,2]
        [(5, 1), (7, 1)],    # B[1,1] + B[1,3]
    ]
    products.append((left6, right6))
    
    # M7 = (A01 - A11)(B10 + B11)
    left7 = [
        [(2, 1), (10, -1)],  # A[0,2] - A[2,2]
        [(3, 1), (11, -1)],  # A[0,3] - A[2,3]
        [(6, 1), (14, -1)],  # A[1,2] - A[3,2]
        [(7, 1), (15, -1)],  # A[1,3] - A[3,3]
    ]
    right7 = [
        [(8, 1), (10, 1)],   # B[2,0] + B[2,2]
        [(9, 1), (11, 1)],   # B[2,1] + B[2,3]
        [(12, 1), (14, 1)],  # B[3,0] + B[3,2]
        [(13, 1), (15, 1)],  # B[3,1] + B[3,3]
    ]
    products.append((left7, right7))
    
    # Build U, V, W for 4x4x4
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for j, (left, right) in enumerate(products):
        for i in range(4):
            for k in range(7):
                # U contribution
                for (idx, coeff) in left[i]:
                    U[idx][j*7 + k] += U2[i][k] * coeff
                # V contribution
                for (idx, coeff) in right[i]:
                    V[idx][j*7 + k] += V2[i][k] * coeff
        # W contribution
        for i in range(4):
            for k in range(7):
                W[i][j*7 + k] += W2[i][k]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # 4x4x4 Strassen part (rank 49)
    U1, V1, W1 = _strassen_4x4()
    
    # 4x4x3 schoolbook part (rank 48)
    rank2 = 4 * 4 * 3  # 48
    U2 = [[0]*rank2 for _ in range(16)]
    V2 = [[0]*rank2 for _ in range(12)]
    W2 = [[0]*rank2 for _ in range(12)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U2[i*4 + j][col] = 1
                V2[j*4 + k][col] = 1
                W2[k*4 + i][col] = 1
                col += 1
    
    # Combine
    total_rank = 49 + rank2  # 97
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Copy Strassen part
    for i in range(16):
        for c in range(49):
            U[i][c] = U1[i][c]
            V[i][c] = V1[i][c]
            W[i][c] = W1[i][c]
    
    # Copy schoolbook part (B2 is rows 4-6 of B, C2 is rows 4-6 of C)
    for i in range(16):
        for c in range(rank2):
            U[i][49 + c] = U2[i][c]
    for j in range(12):
        for c in range(rank2):
            V[16 + j][49 + c] = V2[j][c]
            W[16 + j][49 + c] = W2[j][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
