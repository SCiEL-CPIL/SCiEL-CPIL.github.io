import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # Indices in A (2x3):
    # 0:11, 1:12, 2:13, 3:21, 4:22, 5:23
    
    # For each of the 5 blocks of B (3x2) and C (2x2)
    for i in range(5):
        base_col = i * 11
        b_offset = i * 6
        c_offset = i * 4
        
        # Strassen Part (7 terms)
        # Term 1: M1
        col = base_col
        U[0][col] = 1; U[4][col] = 1
        V[0+b_offset][col] = 1; V[2+b_offset][col] = 1
        W[0+c_offset][col] = 1; W[3+c_offset][col] = 1
        
        # Term 2: M2
        col = base_col + 1
        U[3][col] = 1; U[4][col] = 1
        V[0+b_offset][col] = 1
        W[2+c_offset][col] = 1
        
        # Term 3: M3
        col = base_col + 2
        U[0][col] = 1
        V[1+b_offset][col] = 1; V[3+b_offset][col] = -1
        W[1+c_offset][col] = 1
        
        # Term 4: M4
        col = base_col + 3
        U[4][col] = 1
        V[2+b_offset][col] = 1; V[0+b_offset][col] = -1
        W[0+c_offset][col] = -1; W[2+c_offset][col] = 1
        
        # Term 5: M5
        col = base_col + 4
        U[0][col] = 1; U[1][col] = 1
        V[3+b_offset][col] = 1
        W[0+c_offset][col] = -1; W[1+c_offset][col] = 1
        
        # Term 6: M6
        col = base_col + 5
        U[3][col] = 1; U[0][col] = -1
        V[0+b_offset][col] = 1; V[1+b_offset][col] = 1
        W[3+c_offset][col] = -1
        
        # Term 7: M7
        col = base_col + 6
        U[1][col] = 1; U[4][col] = -1
        V[2+b_offset][col] = 1; V[3+b_offset][col] = 1
        W[0+c_offset][col] = 1
        
        # Cross Product Part (4 terms)
        # a13 * b31 -> c11
        col = base_col + 7
        U[2][col] = 1
        V[4+b_offset][col] = 1
        W[0+c_offset][col] = 1
        
        # a13 * b32 -> c12
        col = base_col + 8
        U[2][col] = 1
        V[5+b_offset][col] = 1
        W[1+c_offset][col] = 1
        
        # a23 * b31 -> c21
        col = base_col + 9
        U[5][col] = 1
        V[4+b_offset][col] = 1
        W[2+c_offset][col] = 1
        
        # a23 * b32 -> c22
        col = base_col + 10
        U[5][col] = 1
        V[5+b_offset][col] = 1
        W[3+c_offset][col] = 1

    return {"U": U, "V": V, "W": W}
