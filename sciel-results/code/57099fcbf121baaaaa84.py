import time
import sys

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                lhs = 0.0
                for r in range(rank):
                    lhs += U[a][r] * V[b][r] * W[c][r]
                if abs(lhs - expected) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # We want to find a rank-11 decomposition.
    # Rank 11 is achievable by combining Strassen (rank 7) for 2x2x2 and schoolbook (rank 4) for 2x2x1.
    # Total rank = 7 + 4 = 11.
    
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Indices:
    # U[i*m+j]
    # V[j*p+k]
    # W[k*n+i]
    
    # Strassen for k=0,1 (columns 0 and 1 of B)
    # M1 = A00 * B00
    # M2 = A10 * (B00 + B01)
    # M3 = (A00 + A01) * B11
    # M4 = A11 * (B10 + B11)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A10 + A11) * (B10 - B00)
    # M7 = (A00 + A01) * (B01 - B10)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    
    # Column 0: M1
    U[0][0] = 1.0  # A00
    V[0][0] = 1.0  # B00
    W[0][0] = 1.0  # C00
    W[3][0] = 1.0  # C11
    
    # Column 1: M2
    U[2][1] = 1.0  # A10
    V[0][1] = 1.0  # B00
    V[1][1] = 1.0  # B01
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11
    
    # Column 2: M3
    U[0][2] = 1.0  # A00
    U[1][2] = 1.0  # A01
    V[4][2] = 1.0  # B11
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11
    
    # Column 3: M4
    U[3][3] = 1.0  # A11
    V[3][3] = 1.0  # B10
    V[4][3] = 1.0  # B11
    W[0][3] = 1.0  # C00
    
    # Column 4: M5
    U[0][4] = 1.0  # A00
    U[3][4] = 1.0  # A11
    V[0][4] = 1.0  # B00
    V[4][4] = 1.0  # B11
    W[0][4] = -1.0 # C00
    W[3][4] = 1.0  # C11
    
    # Column 5: M6
    U[2][5] = 1.0  # A10
    U[3][5] = 1.0  # A11
    V[3][5] = 1.0  # B10
    V[0][5] = -1.0 # B00
    W[2][5] = 1.0  # C10
    
    # Column 6: M7
    U[0][6] = 1.0  # A00
    U[1][6] = 1.0  # A01
    V[1][6] = 1.0  # B01
    V[3][6] = -1.0 # B10
    W[0][6] = -1.0 # C00
    W[1][6] = 1.0  # C01
    
    # Schoolbook for k=2 (column 2 of B)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Column 7: A00 * B02
    U[0][7] = 1.0
    V[2][7] = 1.0  # B02 (0*3+2=2)
    W[4][7] = 1.0  # C02 (2*2+0=4)
    
    # Column 8: A01 * B12
    U[1][8] = 1.0
    V[5][8] = 1.0  # B12 (1*3+2=5)
    W[4][8] = 1.0  # C02
    
    # Column 9: A10 * B02
    U[2][9] = 1.0
    V[2][9] = 1.0
    W[5][9] = 1.0  # C12 (2*2+1=5)
    
    # Column 10: A11 * B12
    U[3][10] = 1.0
    V[5][10] = 1.0
    W[5][10] = 1.0
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 12
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
