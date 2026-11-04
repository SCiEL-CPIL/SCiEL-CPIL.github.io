"""Program for 2x3x6 matrix multiplication tensor decomposition."""

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1 has element A[0][0] at row 0, A[1][0] at row 1
    # B1 has elements B[0][k] for k=0..5
    # C[i][k] = A[i][0] * B[0][k]
    for k in range(p):  # k = 0..5
        for i in range(n):  # i = 0, 1
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    # col is now 12
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2 elements: A[0][1], A[0][2], A[1][1], A[1][2]
    # B2 elements: B[1][k], B[2][k] for k=0..5
    # C[i][k] = A[i][1]*B[1][k] + A[i][2]*B[2][k]
    # Split B2 into three 2x2 blocks: B2_0 (cols 0,1), B2_1 (cols 2,3), B2_2 (cols 4,5)
    # For each block, use Strassen with 7 multiplications
    
    for t in range(3):  # three 2x2 blocks in B2
        # B2 block t has columns 2t, 2t+1
        
        # Strassen M1 = (A00+A11)(B00+B11)
        # A00 = A[0][1], A11 = A[1][2]
        # B00 = B[1][2t], B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M1 contributes to C00 and C11
        W[2*t * n + 0][col] = 1.0  # C00: i=0, k=2t
        W[(2*t + 1) * n + 1][col] = 1.0  # C11: i=1, k=2t+1
        col += 1
        
        # M2 = (A10+A11)B00
        # A10 = A[1][1], A11 = A[1][2]
        # B00 = B[1][2t]
        U[1 * m + 1][col] = 1.0  # A[1][1]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        # M2 contributes to C10 and C11 (with -1)
        W[2*t * n + 1][col] = 1.0  # C10: i=1, k=2t
        W[(2*t + 1) * n + 1][col] = -1.0  # C11: i=1, k=2t+1
        col += 1
        
        # M3 = A00(B01-B11)
        # A00 = A[0][1]
        # B01 = B[1][2t+1], B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        V[1 * p + 2*t + 1][col] = 1.0  # B[1][2t+1]
        V[2 * p + 2*t + 1][col] = -1.0  # B[2][2t+1]
        # M3 contributes to C01 and C11
        W[(2*t + 1) * n + 0][col] = 1.0  # C01: i=0, k=2t+1
        W[(2*t + 1) * n + 1][col] = 1.0  # C11: i=1, k=2t+1
        col += 1
        
        # M4 = A11(B10-B00)
        # A11 = A[1][2]
        # B10 = B[2][2t], B00 = B[1][2t]
        U[1 * m + 2][col] = 1.0  # A[1][2]
        V[2 * p + 2*t][col] = 1.0  # B[2][2t]
        V[1 * p + 2*t][col] = -1.0  # B[1][2t]
        # M4 contributes to C00 and C10
        W[2*t * n + 0][col] = 1.0  # C00: i=0, k=2t
        W[2*t * n + 1][col] = 1.0  # C10: i=1, k=2t
        col += 1
        
        # M5 = (A00+A01)B11
        # A00 = A[0][1], A01 = A[0][2]
        # B11 = B[2][2t+1]
        U[0 * m + 1][col] = 1.0  # A[0][1]
        U[0 * m + 2][col] = 1.0  # A[0][2]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M5 contributes to C00 (with -1) and C01
        W[2*t * n + 0][col] = -1.0  # C00: i=0, k=2t
        W[(2*t + 1) * n + 0][col] = 1.0  # C01: i=0, k=2t+1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 = A[1][1], A00 = A[0][1]
        # B00 = B[1][2t], B01 = B[1][2t+1]
        U[1 * m + 1][col] = 1.0  # A[1][1]
        U[0 * m + 1][col] = -1.0  # A[0][1]
        V[1 * p + 2*t][col] = 1.0  # B[1][2t]
        V[1 * p + 2*t + 1][col] = 1.0  # B[1][2t+1]
        # M6 contributes to C11
        W[(2*t + 1) * n + 1][col] = 1.0  # C11: i=1, k=2t+1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 = A[0][2], A11 = A[1][2]
        # B10 = B[2][2t], B11 = B[2][2t+1]
        U[0 * m + 2][col] = 1.0  # A[0][2]
        U[1 * m + 2][col] = -1.0  # A[1][2]
        V[2 * p + 2*t][col] = 1.0  # B[2][2t]
        V[2 * p + 2*t + 1][col] = 1.0  # B[2][2t+1]
        # M7 contributes to C00
        W[2*t * n + 0][col] = 1.0  # C00: i=0, k=2t
        col += 1
    
    # col should be 12 + 3*7 = 33
    return {"U": U, "V": V, "W": W}
