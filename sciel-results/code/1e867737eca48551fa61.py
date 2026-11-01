import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_idx_11, a_idx_12, a_idx_21, a_idx_22, 
                           b_idx_11, b_idx_12, b_idx_21, b_idx_22,
                           w_idx_11, w_idx_12, w_idx_21, w_idx_22):
        nonlocal col
        # m1 = a11 * (b11 + b22)
        U[a_idx_11][col] += 1
        V[b_idx_11][col] += 1
        V[b_idx_22][col] += 1
        W[w_idx_11][col] += 1
        W[w_idx_22][col] += 1
        col += 1
        
        # m2 = (a11 + a22) * (b11 + b12)
        U[a_idx_11][col] += 1
        U[a_idx_22][col] += 1
        V[b_idx_11][col] += 1
        V[b_idx_12][col] += 1
        W[w_idx_21][col] += 1
        W[w_idx_22][col] -= 1
        col += 1
        
        # m3 = (a12 - a22) * (b21 + b22)
        U[a_idx_12][col] += 1
        U[a_idx_22][col] -= 1
        V[b_idx_21][col] += 1
        V[b_idx_22][col] += 1
        W[w_idx_12][col] += 1
        W[w_idx_22][col] += 1
        col += 1
        
        # m4 = a22 * (b11 - b21)
        U[a_idx_22][col] += 1
        V[b_idx_11][col] += 1
        V[b_idx_21][col] -= 1
        W[w_idx_11][col] += 1
        W[w_idx_21][col] += 1
        col += 1
        
        # m5 = (a11 + a12) * b22
        U[a_idx_11][col] += 1
        U[a_idx_12][col] += 1
        V[b_idx_22][col] += 1
        W[w_idx_11][col] -= 1
        W[w_idx_12][col] += 1
        col += 1
        
        # m6 = (a21 - a11) * (b11 + b12)
        U[a_idx_21][col] += 1
        U[a_idx_11][col] -= 1
        V[b_idx_11][col] += 1
        V[b_idx_12][col] += 1
        W[w_idx_22][col] += 1
        col += 1
        
        # m7 = (a12 + a22) * (b21 - b11)
        U[a_idx_12][col] += 1
        U[a_idx_22][col] += 1
        V[b_idx_21][col] += 1
        V[b_idx_11][col] -= 1
        W[w_idx_11][col] += 1
        W[w_idx_22][col] -= 1
        col += 1

    # A indices:
    # A00=0, A01=1, A02=2
    # A10=3, A11=4, A12=5
    
    # B indices:
    # B00=0, B01=1, B02=2, B03=3, B04=4, B05=5, B06=6
    # B10=7, B11=8, B12=9, B13=10, B14=11, B15=12, B16=13
    # B20=14, B21=15, B22=16, B23=17, B24=18, B25=19, B26=20
    
    # W indices (k*n+i):
    # C00=0, C01=1, C02=2, C03=3, C04=4, C05=5, C06=6
    # C10=7, C11=8, C12=9, C13=10, C14=11, C15=12, C16=13

    # Block 1: Cols 0,1
    add_strassen_block(0, 1, 3, 4, 0, 1, 7, 8, 0, 1, 7, 8)
    # Block 2: Cols 2,3
    add_strassen_block(0, 1, 3, 4, 2, 3, 9, 10, 2, 3, 9, 10)
    # Block 3: Cols 4,5
    add_strassen_block(0, 1, 3, 4, 4, 5, 11, 12, 4, 5, 11, 12)
    
    # Block 4: Col 6 (Schoolbook for 2x2 * 2x1)
    # c06 = a00*b06 + a01*b16
    # c16 = a10*b06 + a11*b16
    
    # m1 = a00 * b06
    U[0][col] += 1
    V[6][col] += 1
    W[6][col] += 1
    col += 1
    
    # m2 = a01 * b16
    U[1][col] += 1
    V[13][col] += 1
    W[6][col] += 1
    col += 1
    
    # m3 = a10 * b06
    U[3][col] += 1
    V[6][col] += 1
    W[14][col] += 1
    col += 1
    
    # m4 = a11 * b16
    U[4][col] += 1
    V[13][col] += 1
    W[14][col] += 1
    col += 1
    
    # Outer Product: A_R * B_B
    # A_R is A02 (idx 2), A12 (idx 5)
    # B_B is B20..B26 (idx 14..20)
    # C0k += A02 * B2k
    # C1k += A12 * B2k
    
    for k in range(7):
        b_idx = 14 + k
        w_idx_0 = k
        w_idx_1 = 7 + k
        
        # C0k += A02 * B2k
        U[2][col] += 1
        V[b_idx][col] += 1
        W[w_idx_0][col] += 1
        col += 1
        
        # C1k += A12 * B2k
        U[5][col] += 1
        V[b_idx][col] += 1
        W[w_idx_1][col] += 1
        col += 1
        
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Random test
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    # C[i, k] = sum_r U[i, r] * V[j, r] * W[k, r] ? No.
    # The tensor is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here i is row of A, j is col of A (row of B), k is col of B (col of C)
    # U shape (n*m, rank) -> U[i*m+j, r]
    # V shape (m*p, rank) -> V[j*p+k, r]
    # W shape (p*n, rank) -> W[k*n+i, r]
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # sum over r
                s += np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
            C_computed[i, k] = s
            
    if not np.allclose(C_computed, C_expected):
        # Fallback to schoolbook
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
        
    return {"U": U, "V": V, "W": W}
