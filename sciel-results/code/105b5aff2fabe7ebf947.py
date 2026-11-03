import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition for 2x2 (Rank 4) - Not used, we use Strassen (Rank 7)
    # Strassen 2x2 decomposition
    # U2, V2, W2 shapes (4, 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # Strassen definitions
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors (A)
    # Indices 0:11, 1:12, 2:21, 3:22
    U2[0, 0] = 1.0; U2[3, 0] = 1.0
    U2[2, 1] = 1.0; U2[3, 1] = 1.0
    U2[0, 2] = 1.0
    U2[3, 3] = 1.0
    U2[0, 4] = 1.0; U2[1, 4] = 1.0
    U2[2, 5] = 1.0; U2[0, 5] = -1.0
    U2[1, 6] = 1.0; U2[3, 6] = -1.0

    # V factors (B)
    # Indices 0:11, 1:12, 2:21, 3:22
    V2[0, 0] = 1.0; V2[3, 0] = 1.0
    V2[0, 1] = 1.0
    V2[2, 2] = 1.0; V2[3, 2] = -1.0
    V2[1, 3] = 1.0; V2[0, 3] = -1.0
    V2[3, 4] = 1.0
    V2[0, 5] = 1.0; V2[1, 5] = 1.0
    V2[2, 6] = 1.0; V2[3, 6] = 1.0

    # W factors (C)
    # Indices 0:11, 1:12, 2:21, 3:22
    W2[0, 0] = 1.0; W2[3, 0] = 1.0; W2[1, 0] = -1.0; W2[2, 0] = 1.0
    W2[1, 1] = 1.0; W2[3, 1] = 1.0
    W2[2, 2] = 1.0; W2[0, 2] = 1.0
    W2[0, 3] = 1.0; W2[2, 3] = -1.0; W2[1, 3] = 1.0; W2[3, 3] = 1.0

    # Trivial 3x3 decomposition (Rank 9)
    U3 = np.zeros((9, 9), dtype=float)
    V3 = np.zeros((9, 9), dtype=float)
    W3 = np.zeros((9, 9), dtype=float)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                idx = i * 3 + k * 0 + j # Wait, index for trivial is usually (i,k,j) or similar
                # Let's just iterate and set
                # U3 row: i*3+k, V3 row: k*3+j, W3 row: j*3+i
                # Rank index: i*3 + k + j*9? No, just unique index for each term
                # Term t corresponds to (i,k,j)
                t = i * 9 + k * 3 + j
                U3[i * 3 + k, t] = 1.0
                V3[k * 3 + j, t] = 1.0
                W3[j * 3 + i, t] = 1.0

    # Construct 6x6 decomposition using Kronecker product
    # Rank = 7 * 9 = 63
    rank = 63
    U6 = np.zeros((36, rank), dtype=float)
    V6 = np.zeros((36, rank), dtype=float)
    W6 = np.zeros((36, rank), dtype=float)

    for r2 in range(7):
        for r3 in range(9):
            r = r2 * 9 + r3
            
            # U6
            for I in range(2):
                for K in range(2):
                    if U2[I, K, r2] != 0: # Wait, U2 is (4,7), index is I*2+K
                        u2_val = U2[I * 2 + K, r2]
                        if u2_val != 0:
                            for i in range(3):
                                for k in range(3):
                                    u3_val = U3[i * 3 + k, r3]
                                    if u3_val != 0:
                                        U6[(I * 3 + i) * 6 + (K * 3 + k), r] += u2_val * u3_val

            # V6
            for K in range(2):
                for J in range(2):
                    v2_val = V2[K * 2 + J, r2]
                    if v2_val != 0:
                        for k in range(3):
                            for j in range(3):
                                v3_val = V3[k * 3 + j, r3]
                                if v3_val != 0:
                                    V6[(K * 3 + k) * 6 + (J * 3 + j), r] += v2_val * v3_val

            # W6
            for J in range(2):
                for I in range(2):
                    w2_val = W2[J * 2 + I, r2]
                    if w2_val != 0:
                        for j in range(3):
                            for i in range(3):
                                w3_val = W3[j * 3 + i, r3]
                                if w3_val != 0:
                                    W6[(J * 3 + j) * 6 + (I * 3 + i), r] += w2_val * w3_val

    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U6, V6, W6)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to trivial if verification fails
        rank_triv = n * m * p
        U_triv = np.zeros((n * m, rank_triv), dtype=float)
        V_triv = np.zeros((m * p, rank_triv), dtype=float)
        W_triv = np.zeros((p * n, rank_triv), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_triv[i * m + k, idx] = 1.0
                    V_triv[k * p + j, idx] = 1.0
                    W_triv[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
