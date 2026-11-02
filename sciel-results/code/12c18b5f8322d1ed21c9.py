import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_S = np.zeros((4, 7))
    V_S = np.zeros((4, 7))
    W_S = np.zeros((4, 7))
    
    # M1 = (A00 - A11)(B10 + B11)
    # A00: i=0,j=0 -> U[0]
    # A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]
    # B11: j=1,k=1 -> V[3]
    # C00: k=0,i=0 -> W[0]
    # C11: k=1,i=1 -> W[3]
    
    U_S[0, 0] = 1
    U_S[3, 0] = -1
    V_S[2, 0] = 1
    V_S[3, 0] = 1
    W_S[0, 0] = 1
    W_S[3, 0] = 1
    
    # M2 = (A00 + A11) B00
    # A00: U[0]
    # A11: U[3]
    # B00: j=0,k=0 -> V[0]
    # C10: k=1,i=0 -> W[2]
    
    U_S[0, 1] = 1
    U_S[3, 1] = 1
    V_S[0, 1] = 1
    W_S[2, 1] = 1
    
    # M3 = A00 (B01 - B11)
    # A00: U[0]
    # B01: j=0,k=1 -> V[1]
    # B11: j=1,k=1 -> V[3]
    # C01: k=0,i=1 -> W[1]
    
    U_S[0, 2] = 1
    V_S[1, 2] = 1
    V_S[3, 2] = -1
    W_S[1, 2] = 1
    
    # M4 = A11 (B10 - B00)
    # A11: U[3]
    # B10: V[2]
    # B00: V[0]
    # C00: W[0]
    
    U_S[3, 3] = 1
    V_S[2, 3] = 1
    V_S[0, 3] = -1
    W_S[0, 3] = 1
    
    # M5 = (A00 + A01) B11
    # A00: U[0]
    # A01: i=0,j=1 -> U[1]
    # B11: V[3]
    # C00: W[0] (with -1)
    # C01: W[1] (with +1)
    
    U_S[0, 4] = 1
    U_S[1, 4] = 1
    V_S[3, 4] = 1
    W_S[0, 4] = -1
    W_S[1, 4] = 1
    
    # M6 = (A10 - A00) (B00 + B01)
    # A10: i=1,j=0 -> U[2]
    # A00: U[0]
    # B00: V[0]
    # B01: V[1]
    # C11: W[3] (with +1)
    
    U_S[2, 5] = 1
    U_S[0, 5] = -1
    V_S[0, 5] = 1
    V_S[1, 5] = 1
    W_S[3, 5] = 1
    
    # M7 = (A01 - A11) (B10 + B11)
    # A01: U[1]
    # A11: U[3]
    # B10: V[2]
    # B11: V[3]
    # C00: W[0] (with +1)
    # C11: W[3] (with -1)
    
    U_S[1, 6] = 1
    U_S[3, 6] = -1
    V_S[2, 6] = 1
    V_S[3, 6] = 1
    W_S[0, 6] = 1
    W_S[3, 6] = -1
    
    # Construct 2x2x4 tensor factors
    # n=2, m=2, p=4
    # U_O: 4x14
    # V_O: 8x14
    # W_O: 8x14
    
    U_O = np.zeros((4, 14))
    V_O = np.zeros((8, 14))
    W_O = np.zeros((8, 14))
    
    # Block 1: k in {0, 1}
    U_O[:, 0:7] = U_S
    # V_O rows for k=0,1:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    V_O[0, 0:7] = V_S[0]
    V_O[1, 0:7] = V_S[1]
    V_O[4, 0:7] = V_S[2]
    V_O[5, 0:7] = V_S[3]
    
    # W_O rows for k=0,1:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    W_O[0, 0:7] = W_S[0]
    W_O[1, 0:7] = W_S[1]
    W_O[2, 0:7] = W_S[2]
    W_O[3, 0:7] = W_S[3]
    
    # Block 2: k in {2, 3}
    U_O[:, 7:14] = U_S
    # V_O rows for k=2,3:
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    V_O[2, 7:14] = V_S[0]
    V_O[3, 7:14] = V_S[1]
    V_O[6, 7:14] = V_S[2]
    V_O[7, 7:14] = V_S[3]
    
    # W_O rows for k=2,3:
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    W_O[4, 7:14] = W_S[0]
    W_O[5, 7:14] = W_S[1]
    W_O[6, 7:14] = W_S[2]
    W_O[7, 7:14] = W_S[3]
    
    # Kronecker products
    # U_raw: 16x98
    # V_raw: 32x98
    # W_raw: 32x98
    
    U_raw = np.kron(U_O, U_S)
    V_raw = np.kron(V_O, V_S)
    W_raw = np.kron(W_O, W_S)
    
    # Permutations
    # U index: i_full = 2*I + i_e, j_full = 2*J + j_e
    # idx = i_full * 4 + j_full = (2I + i_e)*4 + (2J + j_e) = 8I + 4i_e + 2J + j_e
    # Raw index: (I*2 + J)*4 + (i_e*2 + j_e) = 8I + 4J + 2i_e + j_e
    
    perm_U = np.zeros(16, dtype=int)
    for I in range(2):
        for J in range(2):
            for i_e in range(2):
                for j_e in range(2):
                    target = 8*I + 4*i_e + 2*J + j_e
                    source = (I*2 + J)*4 + (i_e*2 + j_e)
                    perm_U[target] = source
                    
    U_final = U_raw[perm_U, :]
    
    # V index: j_full = 2*J + j_e, k_full = 2*K + k_e
    # idx = j_full * 8 + k_full = (2J + j_e)*8 + (2K + k_e) = 16J + 8j_e + 2K + k_e
    # Raw index: (J*4 + K)*4 + (j_e*2 + k_e) = 16J + 4K + 2j_e + k_e
    
    perm_V = np.zeros(32, dtype=int)
    for J in range(2):
        for K in range(4):
            for j_e in range(2):
                for k_e in range(2):
                    target = 16*J + 8*j_e + 2*K + k_e
                    source = (J*4 + K)*4 + (j_e*2 + k_e)
                    perm_V[target] = source
                    
    V_final = V_raw[perm_V, :]
    
    # W index: k_full = 2*K + k_e, i_full = 2*I + i_e
    # idx = k_full * 4 + i_full = (2K + k_e)*4 + (2I + i_e) = 8K + 4k_e + 2I + i_e
    # Raw index: (K*2 + I)*4 + (k_e*2 + i_e) = 8K + 4I + 2k_e + i_e
    
    perm_W = np.zeros(32, dtype=int)
    for K in range(4):
        for I in range(2):
            for k_e in range(2):
                for i_e in range(2):
                    target = 8*K + 4*k_e + 2*I + i_e
                    source = (K*2 + I)*4 + (k_e*2 + i_e)
                    perm_W[target] = source
                    
    W_final = W_raw[perm_W, :]
    
    # Verification
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon = 0.0
                    for r in range(U.shape[1]):
                        recon += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(recon - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U_final, V_final, W_final, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}

    return {"U": [[float(x) for x in row] for row in U_final],
            "V": [[float(x) for x in row] for row in V_final],
            "W": [[float(x) for x in row] for row in W_final]}
