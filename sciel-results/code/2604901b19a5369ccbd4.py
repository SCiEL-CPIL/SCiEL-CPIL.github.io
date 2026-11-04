import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 85 decomposition
    # Part 1: 4x4 by 4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4 by 4x1 multiplication (Rank 16)
    # Part 3: 4x1 by 1x4 multiplication (Rank 16)
    # Part 4: 4x1 by 1x1 multiplication (Rank 4)
    
    total_rank = 49 + 16 + 16 + 4
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # --- Part 1: 4x4 Strassen ---
    # Strassen 2x2 U, V, W
    # Indices 0..3 for 2x2 matrix
    # U: 4x7, V: 4x7, W: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # Coefficients for 4x4 Strassen
    # alpha[k][a][b] for A blocks
    # beta[k][b][c] for B blocks
    # gamma[k][a][c] for C blocks
    
    # k=0 (M1)
    alpha0 = np.array([[1,0],[0,1]]) # A00, A11
    beta0 = np.array([[1,0],[0,1]]) # B00, B11
    gamma0 = np.array([[1,0],[0,1]]) # C00, C11
    
    # k=1 (M2)
    alpha1 = np.array([[0,0],[1,1]]) # A10, A11
    beta1 = np.array([[1,0],[0,0]]) # B00
    gamma1 = np.array([[0,0],[1,-1]]) # C10, C11
    
    # k=2 (M3)
    alpha2 = np.array([[1,0],[0,0]]) # A00
    beta2 = np.array([[0,1],[0,-1]]) # B01, B11
    gamma2 = np.array([[0,1],[0,1]]) # C01, C11
    
    # k=3 (M4)
    alpha3 = np.array([[0,0],[0,1]]) # A11
    beta3 = np.array([[0,-1],[1,0]]) # B00, B10
    gamma3 = np.array([[1,0],[1,0]]) # C00, C10
    
    # k=4 (M5)
    alpha4 = np.array([[1,1],[0,0]]) # A00, A01
    beta4 = np.array([[0,0],[0,1]]) # B11
    gamma4 = np.array([[-1,1],[0,0]]) # C00, C01
    
    # k=5 (M6)
    alpha5 = np.array([[0,0],[1,-1]]) # A10, A00
    beta5 = np.array([[1,1],[0,0]]) # B00, B01
    gamma5 = np.array([[0,0],[0,1]]) # C11
    
    # k=6 (M7)
    alpha6 = np.array([[0,1],[0,-1]]) # A01, A11
    beta6 = np.array([[0,0],[1,1]]) # B10, B11
    gamma6 = np.array([[1,0],[0,0]]) # C00
    
    alphas = [alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6]
    betas = [beta0, beta1, beta2, beta3, beta4, beta5, beta6]
    gammas = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6]
    
    for k in range(7):
        alpha = alphas[k]
        beta = betas[k]
        gamma = gammas[k]
        
        for s in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    coeff = U2[i, s]
                    if coeff != 0:
                        for a in range(2):
                            for b in range(2):
                                val = coeff * alpha[a, b]
                                if val != 0:
                                    u_idx = (2*a + i) * 4 + (2*b + j) # A is 4x4
                                    U[u_idx, col] += val
            # V
            for i in range(2):
                for j in range(2):
                    coeff = V2[i, s]
                    if coeff != 0:
                        for b in range(2):
                            for c in range(2):
                                val = coeff * beta[b, c]
                                if val != 0:
                                    v_idx = (2*b + i) * 4 + (2*c + j) # B is 4x4
                                    V[v_idx, col] += val
            # W
            for i in range(2):
                for j in range(2):
                    coeff = W2[i, s]
                    if coeff != 0:
                        for a in range(2):
                            for c in range(2):
                                val = coeff * gamma[a, c]
                                if val != 0:
                                    w_idx = (2*c + j) * 4 + (2*a + i) # C is 4x4
                                    W[w_idx, col] += val
            col += 1
            
    # --- Part 2: 4x4 by 4x1 ---
    # A_L (4x4) * B_LR (4x1) -> C_2 (4x1)
    # C[i, 4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i # k=4, n=4
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # --- Part 3: 4x1 by 1x4 ---
    # A_R (4x1) * B_RL (1x4) -> C_3 (4x4)
    # C[i,k] = A[i,4] B[4,k]
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 4 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # --- Part 4: 4x1 by 1x1 ---
    # A_R (4x1) * B_RR (1x1) -> C_4 (4x1)
    # C[i,4] = A[i,4] B[4,4]
    for i in range(4):
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 4 + i
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
