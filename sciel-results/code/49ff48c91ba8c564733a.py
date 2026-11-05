import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100

    # 1. Construct 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)

    # 2. Construct 3x3 Schoolbook factors
    # U3: 9x27, V3: 9x27, W3: 9x27
    U3 = np.zeros((9, 27), dtype=np.float64)
    V3 = np.zeros((9, 27), dtype=np.float64)
    W3 = np.zeros((9, 27), dtype=np.float64)
    
    for r in range(3):
        for s in range(3):
            for t in range(3):
                col = r * 9 + s * 3 + t
                U3[r * 3 + s, col] = 1.0
                V3[s * 3 + t, col] = 1.0
                W3[t * 3 + r, col] = 1.0

    # 3. Construct 6x6 factors
    # Rank = 7 * 27 = 189
    rank = 189
    U = np.zeros((36, rank), dtype=np.float64)
    V = np.zeros((36, rank), dtype=np.float64)
    W = np.zeros((36, rank), dtype=np.float64)
    
    # Iterate over all combinations of Strassen and Schoolbook terms
    for c2 in range(7):
        for c3 in range(27):
            col = c2 * 27 + c3
            
            # U: Index (I, J, r, s) -> (I*3+r)*6 + (J*3+s)
            # U2[I*2+J, c2] * U3[r*3+s, c3]
            for I in range(2):
                for J in range(2):
                    u2_val = U2[I * 2 + J, c2]
                    if u2_val == 0: continue
                    for r in range(3):
                        for s in range(3):
                            u3_val = U3[r * 3 + s, c3]
                            if u3_val == 0: continue
                            idx = (I * 3 + r) * 6 + (J * 3 + s)
                            U[idx, col] = u2_val * u3_val

            # V: Index (J, K, s, t) -> (J*3+s)*6 + (K*3+t)
            # V2[J*2+K, c2] * V3[s*3+t, c3]
            for J in range(2):
                for K in range(2):
                    v2_val = V2[J * 2 + K, c2]
                    if v2_val == 0: continue
                    for s in range(3):
                        for t in range(3):
                            v3_val = V3[s * 3 + t, c3]
                            if v3_val == 0: continue
                            idx = (J * 3 + s) * 6 + (K * 3 + t)
                            V[idx, col] = v2_val * v3_val

            # W: Index (K, I, t, r) -> (K*3+t)*6 + (I*3+r)
            # W2[K*2+I, c2] * W3[t*3+r, c3]
            for K in range(2):
                for I in range(2):
                    w2_val = W2[K * 2 + I, c2]
                    if w2_val == 0: continue
                    for t in range(3):
                        for r in range(3):
                            w3_val = W3[t * 3 + r, c3]
                            if w3_val == 0: continue
                            idx = (K * 3 + t) * 6 + (I * 3 + r)
                            W[idx, col] = w2_val * w3_val

    # Optional: Quick Validation
    # if time.time() < deadline:
    #     A = np.random.rand(6, 6)
    #     B = np.random.rand(6, 6)
    #     C_true = A @ B
    #     A_flat = A.flatten()
    #     B_flat = B.flatten()
    #     L = U.T @ A_flat
    #     R = V.T @ B_flat
    #     Prod = L * R
    #     C_decomp_flat = W.T @ Prod
    #     C_decomp = np.zeros((6, 6))
    #     for k in range(6):
    #         for i in range(6):
    #             C_decomp[i, k] = C_decomp_flat[k * 6 + i]
    #     if not np.allclose(C_decomp, C_true):
    #         # Fallback if something went wrong (should not happen)
    #         return {"U": np.zeros((36, 216)).tolist(), "V": np.zeros((36, 216)).tolist(), "W": np.zeros((36, 216)).tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
