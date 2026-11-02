import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's 7-term formula.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention:
      A is 2x2, B is 2x2, C = A @ B
      U[i*2 + j, r] corresponds to A[i,j]
      V[j*2 + k, r] corresponds to B[j,k]
      W[k*2 + i, r] corresponds to C[i,k]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    
    # M2: (A10+A11)B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    
    # M3: A00(B01-B11)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # -B11
    
    # M4: A11(B10-B00)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # -B00
    
    # M5: (A00+A01)B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    
    # M6: (A10-A00)(B00+B01)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # -A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    
    # M7: (A01-A11)(B10+B11)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # -A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    
    # Output W: C[i,k] = sum_r alpha_r * beta_r * W[k*2+i, r]
    # C00 = M1 + M4 - M5 + M7  -> W[0*2+0] = W[0]
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # -M5
    W[0][6] = 1   # M7
    
    # C01 = M3 + M5 -> W[1*2+0] = W[2]
    W[2][2] = 1   # M3
    W[2][4] = 1   # M5
    
    # C10 = M2 + M4 -> W[0*2+1] = W[1]
    W[1][1] = 1   # M2
    W[1][3] = 1   # M4
    
    # C11 = M1 - M2 + M3 + M6 -> W[1*2+1] = W[3]
    W[3][0] = 1   # M1
    W[3][1] = -1  # -M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6
    
    return U, V, W


def _schoolbook_2x2x3():
    """
    Return (U, V, W) for 2x2x3 matrix multiplication using schoolbook (rank 12).
    A is 2x2, B is 2x3, C = A @ B (2x3)
    U shape: (4, 12), V shape: (6, 12), W shape: (6, 12)
    Convention:
      U[i*2 + j, r] for A[i,j]
      V[j*3 + k, r] for B[j,k]
      W[k*2 + i, r] for C[i,k]
    """
    rank = 2 * 2 * 3  # 12
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    
    return U, V, W


def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Compute Kronecker product of two tensor decompositions.
    First: n1 x m1 x p2 (A1 is n1xm1, B1 is m1xp1, C1 is n1xp1)
    Second: n2 x m2 x p2 (A2 is n2xm2, B2 is m2xp2, C2 is n2xp2)
    Result: (n1*n2) x (m1*m2) x (p1*p2)
    
    The Kronecker product A1⊗A2 and B1⊗B2 gives C1⊗C2.
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    # U shape: (n1*n2 * m1*m2, rank)
    # Index in U: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
    # V shape: (m1*m2 * p1*p2, rank)
    # Index in V: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
    # W shape: (p1*p2 * n1*n2, rank)
    # Index in W: (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
    
    U = [[0]*rank for _ in range(n1*n2 * m1*m2)]
    V = [[0]*rank for _ in range(m1*m2 * p1*p2)]
    W = [[0]*rank for _ in range(p1*p2 * n1*n2)]
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1 + j1][r1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2*m2 + j2][r2]
                            if u2_val == 0:
                                continue
                            u_idx = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                            U[u_idx][r] = u1_val * u2_val
            
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1*p1 + k1][r1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2*p2 + k2][r2]
                            if v2_val == 0:
                                continue
                            v_idx = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                            V[v_idx][r] = v1_val * v2_val
            
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1*n1 + i1][r1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2*n2 + i2][r2]
                            if w2_val == 0:
                                continue
                            w_idx = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                            W[w_idx][r] = w1_val * w2_val
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s margin
    
    n, m, p = 4, 4, 6
    
    # Strategy: Kronecker product of Strassen 2x2x2 (rank 7) and schoolbook 2x2x3 (rank 12)
    # This gives (2*2)x(2*2)x(2*3) = 4x4x6 with rank 7*12 = 84
    
    U1, V1, W1 = _strassen_2x2x2()  # 2x2x2, rank 7
    U2, V2, W2 = _schoolbook_2x2x3()  # 2x2x3, rank 12
    
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 
                               n1=2, m1=2, p1=2, 
                               n2=2, m2=2, p2=3)
    
    rank = len(U[0])
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = np.dot(U_np.flatten(), A_test.flatten())
        beta_r = np.dot(V_np.flatten(), B_test.flatten())
        if alpha_r != 0 and beta_r != 0:
            C_computed += alpha_r * beta_r * W_np.reshape(6, 4).T  # W[k*n+i, r] -> reshape to (p, n) then transpose
    
    # Actually, let's be more careful with the verification
    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = 0.0
        for i in range(4):
            for j in range(4):
                alpha_r += U_np[i*4 + j, r] * A_test[i, j]
        beta_r = 0.0
        for j in range(4):
            for k in range(6):
                beta_r += V_np[j*6 + k, r] * B_test[j, k]
        for i in range(4):
            for k in range(6):
                C_computed[i, k] += alpha_r * beta_r * W_np[k*4 + i, r]
    
    error = np.max(np.abs(C_computed - C_expected))
    
    if error > 1e-10:
        # Fallback to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}
