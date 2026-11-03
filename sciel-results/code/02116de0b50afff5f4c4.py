import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # Split n=3 into 1+2, m=4 into 2+2, p=5 into 2+3
    # C = A11*B11 + A12*B21 + A21*B12 + A22*B22
    # where:
    #   A11 is 1x2, B11 is 2x2 -> C11 is 1x2
    #   A12 is 1x2, B21 is 2x3 -> C12 is 1x3
    #   A21 is 2x2, B12 is 2x2 -> C21 is 2x2
    #   A22 is 2x2, B22 is 2x3 -> C22 is 2x3
    
    # For each block, we need a tensor decomposition.
    # We'll use schoolbook for most, and Strassen for the 2x2x2 block.
    
    # Helper: create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper: create Strassen decomposition for 2x2x2
    # Strassen: 7 terms instead of 8
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # U[i*2+j, r] for i,j in {0,1}
        # V[j*2+k, r] for j,k in {0,1}
        # W[k*2+i, r] for k,i in {0,1}
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is at U[0*2+0] = U[0], A22 is at U[1*2+1] = U[3]
        # B11 is at V[0*2+0] = V[0], B22 is at V[1*2+1] = V[3]
        # C11 = W[0*2+0] = W[0], C12 = W[0*2+1] = W[1]
        # C21 = W[1*2+0] = W[2], C22 = W[1*2+1] = W[3]
        
        # Term 0: M1
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11
        W[3][0] = 1  # C22
        
        # Term 1: M2 = (A21+A22)B11
        U[2][1] = 1  # A21 (row 1, col 0) -> index 1*2+0 = 2
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21
        W[3][1] = 1  # C22
        
        # Term 2: M3 = A11(B21-B22)
        U[0][2] = 1  # A11
        V[1][2] = 1  # B21 (row 1, col 0) -> index 1*2+0 = 2... wait
        # B is 2x2, B[j,k]. B21 means j=1, k=0? No, B21 is row 1, col 1? 
        # Let me be careful: B is 2x2, indexed B[j,k] where j is row (0,1), k is col (0,1)
        # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
        # So B21 is at V[1*2+0] = V[2], B22 is at V[1*2+1] = V[3]
        V[2][2] = 1  # B21
        V[3][2] = -1 # B22
        W[1][2] = 1  # C12
        W[3][2] = 1  # C22
        
        # Term 3: M4 = A22(B12-B11)
        U[3][3] = 1  # A22
        V[1][3] = 1  # B12 = B[0,1] -> V[0*2+1] = V[1]
        V[0][3] = -1 # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # Term 4: M5 = (A11+A12)B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12 = A[0,1] -> U[0*2+1] = U[1]
        V[3][4] = 1  # B22
        W[0][4] = -1 # C11
        W[1][4] = 1  # C12
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1 # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1 # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        W[1][6] = 1  # C12
        
        return U, V, W, 7
    
    # Now build the full decomposition
    # Block 1: A11 (1x2) * B11 (2x2) -> C11 (1x2)
    # In the full matrix:
    #   A11 corresponds to rows 0, cols 0-1 of A (3x4)
    #   B11 corresponds to rows 0-1, cols 0-1 of B (4x5)
    #   C11 corresponds to row 0, cols 0-1 of C (3x5)
    
    # Block 2: A12 (1x2) * B21 (2x3) -> C12 (1x3)
    #   A12: rows 0, cols 2-3 of A
    #   B21: rows 2-3, cols 0-2 of B
    #   C12: row 0, cols 2-4 of C
    
    # Block 3: A21 (2x2) * B12 (2x2) -> C21 (2x2)
    #   A21: rows 1-2, cols 0-1 of A
    #   B12: rows 0-1, cols 2-3 of B
    #   C21: rows 1-2, cols 2-3 of C
    
    # Block 4: A22 (2x2) * B22 (2x3) -> C22 (2x3)
    #   A22: rows 1-2, cols 2-3 of A
    #   B22: rows 2-3, cols 2-4 of B
    #   C22: rows 1-2, cols 2-4 of C
    
    # Let me use schoolbook for blocks 1, 2, 4 and Strassen for block 3
    # Block 1: 1x2x2 -> schoolbook rank 4
    # Block 2: 1x2x3 -> schoolbook rank 6
    # Block 3: 2x2x2 -> Strassen rank 7
    # Block 4: 2x2x3 -> schoolbook rank 12
    # Total: 4+6+7+12 = 29
    
    U_full = [[0]*29 for _ in range(12)]  # 3*4 = 12
    V_full = [[0]*29 for _ in range(20)]  # 4*5 = 20
    W_full = [[0]*29 for _ in range(15)]  # 5*3 = 15
    
    col_offset = 0
    
    # Block 1: A11 (1x2) * B11 (2x2) -> C11 (1x2)
    # A11: A[0,0], A[0,1] -> U indices 0, 1
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,2,3
    # C11: C[0,0], C[0,1] -> W indices 0, 1 (W[k*n+i] = W[0*3+0]=W[0], W[1*3+0]=W[3])
    # Wait, W[k*n+i], so C[0,0] -> W[0*3+0] = W[0], C[0,1] -> W[1*3+0] = W[3]
    
    U1, V1, W1, rank1 = schoolbook(1, 2, 2)
    for i in range(4):
        for r in range(rank1):
            U_full[i][col_offset + r] = U1[i][r]
    for j in range(4):
        for r in range(rank1):
            V_full[j][col_offset + r] = V1[j][r]
    for k in range(2):
        for r in range(rank1):
            W_full[k*3 + 0][col_offset + r] = W1[k*1 + 0][r]  # C[0,k] -> W[k*3+0]
    col_offset += rank1
    
    # Block 2: A12 (1x2) * B21 (2x3) -> C12 (1x3)
    # A12: A[0,2], A[0,3] -> U indices 2, 3
    # B21: B[2,0], B[2,1], B[2,2], B[3,0], B[3,1], B[3,2] -> V indices 10,11,12,15,16,17
    # C12: C[0,2], C[0,3], C[0,4] -> W indices 6, 9, 12 (W[k*3+0] for k=2,3,4)
    
    U2, V2, W2, rank2 = schoolbook(1, 2, 3)
    for i in range(4):
        for r in range(rank2):
            U_full[2 + i][col_offset + r] = U2[i][r]  # A[0, 2+i] -> U[2+i]
    for j in range(6):
        for r in range(rank2):
            # V2[j] corresponds to B[j//3, j%3] in the local 2x3 matrix
            # B21 is B[2:4, 0:3], so local (j_local, k_local) -> global (2+j_local, k_local)
            # V index = (2+j_local)*5 + k_local = 10 + j_local*5 + k_local
            j_local = j // 3
            k_local = j % 3
            V_full[10 + j_local*5 + k_local][col_offset + r] = V2[j][r]
    for k in range(3):
        for r in range(rank2):
            # C12[k] = C[0, 2+k] -> W[(2+k)*3 + 0] = W[6+k*3]
            W_full[6 + k*3][col_offset + r] = W2[k*1 + 0][r]
    col_offset += rank2
    
    # Block 3: A21 (2x2) * B12 (2x2) -> C21 (2x2) using Strassen
    # A21: A[1,0], A[1,1], A[2,0], A[2,1] -> U indices 4,5,8,9
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2,3,7,8
    # C21: C[1,2], C[1,3], C[2,2], C[2,3] -> W indices 6,9,12,15
    
    U3, V3, W3, rank3 = strassen_2x2x2()
    # Map local indices to global
    # Local U: [A11, A12, A21, A22] -> global U: [4,5,8,9]
    local_to_global_U = [4, 5, 8, 9]
    for i_local in range(4):
        for r in range(rank3):
            U_full[local_to_global_U[i_local]][col_offset + r] = U3[i_local][r]
    
    # Local V: [B11, B12, B21, B22] -> global V: [2,3,7,8]
    local_to_global_V = [2, 3, 7, 8]
    for j_local in range(4):
        for r in range(rank3):
            V_full[local_to_global_V[j_local]][col_offset + r] = V3[j_local][r]
    
    # Local W: [C11, C12, C21, C22] -> global W: [6,9,12,15]
    # W[k*2+i] for k,i in {0,1}
    # C11 = C[1,2] -> W[2*3+1] = W[7]... wait
    # C is 3x5, C[i,k]. C21 is C[1:3, 2:4], so C[1,2], C[1,3], C[2,2], C[2,3]
    # W[k*n+i] = W[k*3+i]
    # C[1,2] -> W[2*3+1] = W[7]
    # C[1,3] -> W[3*3+1] = W[10]
    # C[2,2] -> W[2*3+2] = W[8]
    # C[2,3] -> W[3*3+2] = W[11]
    local_to_global_W = [7, 10, 8, 11]
    for k_local in range(4):
        for r in range(rank3):
            W_full[local_to_global_W[k_local]][col_offset + r] = W3[k_local][r]
    col_offset += rank3
    
    # Block 4: A22 (2x2) * B22 (2x3) -> C22 (2x3)
    # A22: A[1,2], A[1,3], A[2,2], A[2,3] -> U indices 6,7,10,11
    # B22: B[2,2], B[2,3], B[2,4], B[3,2], B[3,3], B[3,4] -> V indices 12,13,14,17,18,19
    # C22: C[1,2], C[1,3], C[1,4], C[2,2], C[2,3], C[2,4] -> W indices 7,10,13,8,11,14
    
    U4, V4, W4, rank4 = schoolbook(2, 2, 3)
    local_to_global_U4 = [6, 7, 10, 11]
    for i_local in range(4):
        for r in range(rank4):
            U_full[local_to_global_U4[i_local]][col_offset + r] = U4[i_local][r]
    
    local_to_global_V4 = [12, 13, 14, 17, 18, 19]
    for j_local in range(6):
        for r in range(rank4):
            V_full[local_to_global_V4[j_local]][col_offset + r] = V4[j_local][r]
    
    local_to_global_W4 = [7, 10, 13, 8, 11, 14]
    for k_local in range(6):
        for r in range(rank4):
            W_full[local_to_global_W4[k_local]][col_offset + r] = W4[k_local][r]
    col_offset += rank4
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    # C[i,k] += sum_j A[i,j]*U[i*m+j,r] * sum_j B[j,k]*V[j*p+k,r] * W[k*n+i,r]
                    a_term = sum(A[i, j] * U[i*m+j][r] for j in range(m))
                    b_term = sum(B[j, k] * V[j*p+k][r] for j in range(m))
                    C_computed[i, k] += a_term * b_term * W[k*n+i][r]
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook
        U_s, V_s, W_s, _ = schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return result

