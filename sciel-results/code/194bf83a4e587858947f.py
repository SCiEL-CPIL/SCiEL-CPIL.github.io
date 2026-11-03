"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition.
Uses recursive Strassen construction to achieve rank-98 decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    t0 = time.time()
    deadline = 120.0 - 5.0  # safety margin
    
    # Strassen 2x2 rank-7 factors
    # U2[i] corresponds to the i-th product's A-factor (flattened 2x2)
    # V2[i] corresponds to the i-th product's B-factor (flattened 2x2)
    # W2[i] corresponds to the i-th product's C-factor (flattened 2x2)
    
    U2 = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],   # C00+C11
        [0, 0, 1, -1],  # C10-C11
        [0, 1, 0, 1],   # C01+C11
        [1, 0, 1, 0],   # C00+C10
        [-1, 1, 0, 0],  # -C00+C01
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0]    # C00
    ], dtype=float)
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def idx2(i, j):
        return i * 2 + j
    
    # Build 4x4x4 rank-49 factors using Kronecker-like composition of Strassen
    for s in range(7):
        for t in range(7):
            # U4: 4x4 matrix flattened to 16 entries
            # U4[I,J] = U2[s][block_row, block_col] * U2[t][local_row, local_col]
            U4 = np.zeros(16)
            V4 = np.zeros(16)
            W4 = np.zeros(16)
            
            for I in range(4):
                for J in range(4):
                    br = I // 2
                    bc = J // 2
                    lr = I % 2
                    lc = J % 2
                    U4[I * 4 + J] = U2[s, idx2(br, bc)] * U2[t, idx2(lr, lc)]
            
            for J in range(4):
                for K in range(4):
                    br = J // 2
                    bc = K // 2
                    lr = J % 2
                    lc = K % 2
                    V4[J * 4 + K] = V2[s, idx2(br, bc)] * V2[t, idx2(lr, lc)]
            
            for I in range(4):
                for K in range(4):
                    br = I // 2
                    bc = K // 2
                    lr = I % 2
                    lc = K % 2
                    W4[I * 4 + K] = W2[s, idx2(br, bc)] * W2[t, idx2(lr, lc)]
            
            # Left block: columns 0-3 of the 8-column output
            col_l = s * 7 + t
            U[:, col_l] = U4
            for j in range(4):
                for k in range(4):
                    V[j * 8 + k, col_l] = V4[j * 4 + k]
            for k in range(4):
                for i in range(4):
                    W[k * 4 + i, col_l] = W4[i * 4 + k]
            
            # Right block: columns 4-7 of the 8-column output
            col_r = 49 + s * 7 + t
            U[:, col_r] = U4
            for j in range(4):
                for k in range(4):
                    V[j * 8 + (k + 4), col_r] = V4[j * 4 + k]
            for k in range(4):
                for i in range(4):
                    W[(k + 4) * 4 + i, col_r] = W4[i * 4 + k]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
