import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def construct():
    n, m, p = 4, 4, 7
    
    # Inner Strassen 2x2x2 coefficients
    # U_inner[s], V_inner[s], W_inner[s] are length 4
    U_inner = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V_inner = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W_inner = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Outer Strassen 2x2x2 block coefficients
    # c[t], d[t], w[t] are 2x2 matrices
    c_outer = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [-1, 0]]  # M7
    ]
    d_outer = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    w_outer = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]], # M3
        [[1, 0], [1, 0]], # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Fill 4x4x4 part
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # U
            for I in range(4):
                for J in range(4):
                    x = I // 2
                    y = J // 2
                    i = I % 2
                    j = J % 2
                    U[I * 4 + J, r] = c_outer[t][x][y] * U_inner[s][i * 2 + j]
            
            # V
            for K in range(4):
                for L in range(4):
                    u = K // 2
                    v = L // 2
                    k = K % 2
                    l = L % 2
                    V[K * 7 + L, r] = d_outer[t][u][v] * V_inner[s][k * 2 + l]
            
            # W
            for Ip in range(4):
                for Kp in range(4):
                    xp = Ip // 2
                    yp = Kp // 2
                    ip = Ip % 2
                    kp = Kp % 2
                    W[Kp * 4 + Ip, r] = w_outer[t][xp][yp] * W_inner[s][ip * 2 + kp]
    
    # Fill 4x4x3 part
    r = rank_444
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, r] = 1
                V[j * 7 + k, r] = 1
                W[k * 4 + i, r] = 1
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
