"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_strassen_blocks():
    """
    Structural approach: Use Strassen's algorithm on the 2x2 block structure 
    of the matrix multiplication, then handle the resulting 1x8 multiplications 
    with schoolbook (which is optimal for 1xn).
    
    A (2x4) is partitioned into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    B (4x8) is partitioned into 2x8 blocks: B = [[B11], [B22]] where B11, B22 are 2x8
    
    Wait, standard block partitioning for A(2x4) and B(4x8):
    A = [A_left, A_right] where A_left, A_right are 2x2
    B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    
    C = A * B = A_left * B_top + A_right * B_bottom
    
    This is just two independent 2x2x8 multiplications.
    Using Strassen on each 2x2x8:
    For 2x2x8, we have 7 "scalar" multiplications, but each scalar is actually a 1x8 vector.
    So we need 7 * 8 = 56 multiplications per 2x2x8 block.
    Total rank = 56 + 56 = 112. This is WORSE than schoolbook (64).
    
    Let's try a different partition.
    A (2x4) -> 2 rows of length 4.
    B (4x8) -> 4 rows of length 8.
    
    Maybe partition the middle dimension (4) differently?
    Or partition the output dimension (8)?
    
    Let's try partitioning the output k into two groups of 4.
    C1 (2x4) = A * B1 (where B1 is 4x4)
    C2 (2x4) = A * B2 (where B2 is 4x4)
    
    This is two independent 2x4x4 multiplications.
    Schoolbook rank for 2x4x4 is 2*4*4 = 32.
    Total rank = 32 + 32 = 64. Same as schoolbook.
    
    Can we do better for 2x4x4?
    Known results: R(2,4,4) is likely 32 (schoolbook) or maybe lower?
    Actually, for 2x2x2, Strassen gives rank 7 < 8.
    For 2x4x4, if we view it as two 2x2x4 blocks?
    Or 2x2x2 blocks?
    
    Let's try to decompose 2x4x4 using Strassen on the 2x2 sub-blocks.
    A (2x4) = [A1, A2] where A1, A2 are 2x2.
    B (4x4) = [B1; B2] where B1, B2 are 2x2.
    C (2x4) = [C1, C2] where C1 = A1*B1 + A2*B2? No.
    C = A * B = [A1, A2] * [B1; B2] = A1*B1 + A2*B2.
    Here B1 is 2x2, B2 is 2x2.
    So C1 (2x2) = A1*B1 and C2 (2x2) = A2*B2?
    Wait, dimensions:
    A1 (2x2) * B1 (2x2) -> 2x2
    A2 (2x2) * B2 (2x2) -> 2x2
    So C = [A1*B1, A2*B2] (horizontal concatenation).
    This is two independent 2x2x2 multiplications.
    Rank of 2x2x2 is 7 (Strassen).
    Total rank = 7 + 7 = 14.
    Schoolbook for 2x4x4 is 32.
    So R(2,4,4) <= 14?
    Let's check the tensor decomposition.
    T[i, j, k] for i in 0..1, j in 0..3, k in 0..3.
    If we split j into j1 (0..1) and j2 (2..3), and k into k1 (0..1) and k2 (2..3).
    C[i, k1] = sum_{j1} A[i, j1] B[j1, k1]
    C[i, k2] = sum_{j2} A[i, j2] B[j2, k2]
    These are independent.
    So yes, we can use Strassen for the first part (i, j1, k1) and Strassen for the second part (i, j2, k2).
    Rank = 7 + 7 = 14.
    
    Now back to 2x4x8.
    Split k into two groups of 4: k1 (0..3), k2 (4..7).
    C1 (2x4) = A * B1 (B1 is 4x4)
    C2 (2x4) = A * B2 (B2 is 4x4)
    Using the 14-rank decomposition for each 2x4x4:
    Total rank = 14 + 14 = 28.
    
    Let's verify this.
    For C1:
    C1[i, k1] = sum_{j=0}^3 A[i, j] B1[j, k1]
    We split j into j1 (0,1) and j2 (2,3).
    C1[i, k1] = sum_{j1=0}^1 A[i, j1] B1[j1, k1] + sum_{j2=2}^3 A[i, j2] B1[j2, k1]
    Wait, this is not correct.
    C1[i, k1] = sum_{j=0}^3 A[i, j] B1[j, k1].
    If we split j, we have two terms.
    But Strassen works on 2x2 matrices.
    Here, for fixed k1, we are computing a vector of length 2 (i=0,1) from a vector of length 4 (j=0..3) and a vector of length 4 (j=0..3, for B).
    Actually, for fixed k1, B1[:, k1] is a 4x1 vector.
    So C1[:, k1] = A * B1[:, k1].
    This is a 2x4 * 4x1 multiplication.
    This is not a 2x2x2 multiplication.
    
    My previous decomposition of 2x4x4 was:
    C (2x4) = [C_left, C_right]
    C_left (2x2) = A_left (2x2) * B_top (2x2)
    C_right (2x2) = A_right (2x2) * B_bottom (2x2)
    This requires B to be partitioned as [B_top; B_bottom] where B_top is 2x2 and B_bottom is 2x2.
    And A partitioned as [A_left, A_right].
    Then C = A * B = [A_left, A_right] * [B_top; B_bottom] = A_left*B_top + A_right*B_bottom.
    This is a SUM of two 2x2 matrices, not a concatenation.
    So C is 2x2, not 2x4.
    
    Ah, the dimensions don't match for a simple block decomposition of 2x4x4 into 2x2x2.
    A is 2x4, B is 4x4, C is 2x4.
    If we split A into [A1, A2] (2x2 each) and B into [B1; B2] (2x2 each, but B is 4x4, so B1 is 2x4, B2 is 2x4).
    Then C = A1*B1 + A2*B2.
    B1 is 2x4, B2 is 2x4.
    So we have two 2x2x4 multiplications.
    Rank of 2x2x4?
    Split k in B into two groups of 2.
    B1 = [B11, B12] (2x2 each).
    A1*B1 = A1*B11 + A1*B12.
    This is two 2x2x2 multiplications.
    Rank 7 + 7 = 14.
    So A1*B1 has rank 14.
    Similarly A2*B2 has rank 14.
    Total rank for 2x4x4 is 14 + 14 = 28.
    Schoolbook is 32.
    So R(2,4,4) <= 28.
    
    Now for 2x4x8:
    Split k into two groups of 4.
    C1 (2x4) = A * B1 (B1 is 4x4). Rank 28.
    C2 (2x4) = A * B2 (B2 is 4x4). Rank 28.
    Total rank = 28 + 28 = 56.
    
    This is an improvement over schoolbook (64).
    Let's implement this.
    
    We need to construct the U, V, W matrices for this decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We will build the decomposition by combining smaller decompositions.
    # Base case: 2x2x2 Strassen decomposition.
    # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Strassen formulas for 2x2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b11)
    # m4 = a22(b12-b22)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)b12
    # m7 = (a12-a22)b21
    # c11 = m1 - m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m3 - m2 + m6
    
    # Let's define the Strassen factors for 2x2x2.
    # U rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # V rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # W rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # c=0: m1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    # Wait, W is indexed by [k*n+i, c] = [k*2+i, c].
    # c11 gets m1. c11 is i=0, k=0. W[0, 0] += 1.
    # c12 gets -m1? No, c12 = m3 + m5.
    # Let's re-evaluate the W coefficients.
    # c11 = m1 - m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m3 - m2 + m6
    
    # Let's fill W_s based on these coefficients.
    # W_s[k*2+i, c]
    
    # c=0 (m1)
    # c11: +1 -> W_s[0, 0] = 1
    # c22: +1 -> W_s[3, 0] = 1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # c=1 (m2)
    # c21: +1 -> W_s[1, 1] = 1 (C[1,0] is i=1, k=0 -> index 1)
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1
    
    # c=2 (m3)
    # c12: +1 -> W_s[2, 2] = 1 (C[0,1] is i=0, k=1 -> index 2)
    # c22: -1 -> W_s[3, 2] = -1 (C[1,1] is i=1, k=1 -> index 3)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[0, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = -1
    
    # c=3 (m4)
    # c11: -1 -> W_s[0, 3] = -1 (C[0,0] is i=0, k=0 -> index 0)
    # c21: +1 -> W_s[1, 3] = 1 (C[1,0] is i=1, k=0 -> index 1)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[3, 3] = -1
    W_s[0, 3] = -1; W_s[1, 3] = 1
    
    # c=4 (m5)
    # c11: -1 -> W_s[0, 4] = -1 (C[0,0] is i=0, k=0 -> index 0)
    # c12: +1 -> W_s[2, 4] = 1 (C[0,1] is i=0, k=1 -> index 2)
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # c=5 (m6)
    # c22: +1 -> W_s[3, 5] = 1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # c=6 (m7)
    # c11: +1 -> W_s[0, 6] = 1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1
    W_s[0, 6] = 1
    
    # Verify Strassen 2x2x2
    def _verify_2x2x2(U, V, W):
        n, m, p = 2, 2, 2
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for c in range(7):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify_2x2x2(U_s, V_s, W_s):
        # Fallback to schoolbook if Strassen is wrong
        return _schoolbook()
    
    # Now build 2x2x4 from two 2x2x2.
    # A (2x2), B (2x4) = [B1, B2] (2x2 each).
    # C (2x2) = A*B1 + A*B2.
    # This is sum of two 2x2x2 products.
    # We can concatenate the rank-1 tensors.
    # Rank = 7 + 7 = 14.
    
    def _combine_add(U1, V1, W1, U2, V2, W2):
        # U1 (n*m, r1), V1 (m*p1, r1), W1 (p1*n, r1)
        # U2 (n*m, r2), V2 (m*p2, r2), W2 (p2*n, r2)
        # Result: U (n*m, r1+r2), V (m*(p1+p2), r1+r2), W ((p1+p2)*n, r1+r2)
        # This assumes the p dimensions are concatenated.
        # But here p1=2, p2=2, total p=4.
        # V1 is (2*2, 7), V2 is (2*2, 7).
        # V_total should be (2*4, 14).
        # V_total[j*4+k, c]
        # If k < 2, V_total[j*4+k, c] = V1[j*2+k, c]
        # If k >= 2, V_total[j*4+k, c] = V2[j*2+(k-2), c+7]
        
        n, m = U1.shape[0] // 2, 2 # This is getting complicated to infer dimensions.
        # Let's pass dimensions explicitly.
        pass
    
    # Let's do it manually for 2x2x4.
    # n=2, m=2, p=4.
    # U_2x2x4 (4, 14)
    # V_2x2x4 (8, 14)
    # W_2x2x4 (8, 14)
    
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    # First part (k=0,1)
    U_224[:, 0:7] = U_s
    for j in range(2):
        for k in range(2):
            V_224[j*4+k, 0:7] = V_s[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[k*2+i, 0:7] = W_s[k*2+i, 0:7]
            
    # Second part (k=2,3)
    U_224[:, 7:14] = U_s
    for j in range(2):
        for k in range(2):
            V_224[j*4+(k+2), 7:14] = V_s[j*2+k, 0:7]
    for k in range(2):
        for i in range(2):
            W_224[(k+2)*2+i, 7:14] = W_s[k*2+i, 0:7]
            
    # Verify 2x2x4
    def _verify_2x2x4(U, V, W):
        n, m, p = 2, 2, 4
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for c in range(14):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify_2x2x4(U_224, V_224, W_224):
        return _schoolbook()
        
    # Now build 2x4x4 from two 2x2x4.
    # A (2x4) = [A1, A2] (2x2 each).
    # B (4x4) = [B1; B2] (2x4 each).
    # C (2x4) = A1*B1 + A2*B2.
    # This is sum of two 2x2x4 products.
    # Rank = 14 + 14 = 28.
    
    U_244 = np.zeros((8, 28))
    V_244 = np.zeros((16, 28))
    W_244 = np.zeros((8, 28))
    
    # First part (j=0,1)
    U_244[0:4, 0:14] = U_224
    for j in range(2):
        for k in range(4):
            V_244[j*4+k, 0:14] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 0:14] = W_224[k*2+i, 0:14]
            
    # Second part (j=2,3)
    U_244[4:8, 14:28] = U_224
    for j in range(2):
        for k in range(4):
            V_244[(j+2)*4+k, 14:28] = V_224[j*4+k, 0:14]
    for k in range(4):
        for i in range(2):
            W_244[k*2+i, 14:28] = W_224[k*2+i, 0:14]
            
    # Verify 2x4x4
    def _verify_2x4x4(U, V, W):
        n, m, p = 2, 4, 4
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for c in range(28):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify_2x4x4(U_244, V_244, W_244):
        return _schoolbook()
        
    # Now build 2x4x8 from two 2x4x4.
    # A (2x4).
    # B (4x8) = [B1, B2] (4x4 each).
    # C (2x8) = [A*B1, A*B2].
    # This is concatenation of two 2x4x4 products.
    # Rank = 28 + 28 = 56.
    
    U_248 = np.zeros((8, 56))
    V_248 = np.zeros((32, 56))
    W_248 = np.zeros((16, 56))
    
    # First part (k=0..3)
    U_248[:, 0:28] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+k, 0:28] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[k*2+i, 0:28] = W_244[k*2+i, 0:28]
            
    # Second part (k=4..7)
    U_248[:, 28:56] = U_244
    for j in range(4):
        for k in range(4):
            V_248[j*8+(k+4), 28:56] = V_244[j*4+k, 0:28]
    for k in range(4):
        for i in range(2):
            W_248[(k+4)*2+i, 28:56] = W_244[k*2+i, 0:28]
            
    # Verify 2x4x8
    def _verify_2x4x8(U, V, W):
        n, m, p = 2, 4, 8
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for c in range(56):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not _verify_2x4x8(U_248, V_248, W_248):
        return _schoolbook()
        
    # Convert to lists
    U = [[int(x) for x in row] for row in U_248]
    V = [[int(x) for x in row] for row in V_248]
    W = [[int(x) for x in row] for row in W_248]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try the Strassen-based construction
    strassen_result = _construct_strassen_blocks()
    strassen_rank = len(strassen_result["U"][0])
    
    if strassen_rank < best_rank:
        if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
            best = strassen_result
            best_rank = strassen_rank
    
    return best
# EVOLVE-BLOCK-END
