import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 32  # 16 terms per half of the 8 columns
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Schoolbook decomposition for 4x4x8
    # For each half (0 or 1), and each (i, j, k) where k is in [half*4, half*4+4)
    # C[i, k] += A[i, j] * B[j, k]
    # 
    # U index: i*4 + j (row i, col j of A, 4x4)
    # V index: j*8 + k (row j, col k of B, 4x8)
    # W index: k*4 + i (row k, col i of C, 8x4) -- per trace convention W[k*n+i, r]
    
    col = 0
    for half in range(2):
        for i in range(4):
            for j in range(4):
                for k_offset in range(4):
                    k = half * 4 + k_offset
                    
                    # U: A[i,j]
                    u_idx = i * 4 + j
                    U[u_idx, col] = 1
                    
                    # V: B[j,k]
                    v_idx = j * 8 + k
                    V[v_idx, col] = 1
                    
                    # W: C[i,k] -> W[k*4 + i]
                    w_idx = k * 4 + i
                    W[w_idx, col] = 1
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
