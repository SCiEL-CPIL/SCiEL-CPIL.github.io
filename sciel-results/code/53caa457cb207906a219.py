import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook rank 128
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    try:
        # 2x2x2 Strassen decomposition (rank 7)
        # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V2 = np.array([
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, -1, 0, 0, 0],
            [1, 0, -1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1]
        ])
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        
        # Verify 2x2x2: sum_r U2[i*2+j,r]*V2[j*2+k,r]*W2[k*2+i,r] == 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] for r in range(7))
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if not valid:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
        # Build 4x4x4 via Kronecker: rank 49
        U4 = np.kron(U2, U2)  # (16, 49)
        V4 = np.kron(V2, V2)  # (16, 49)
        W4 = np.kron(W2, W2)  # (16, 49)
        
        # Quick verify 4x4x4
        valid4 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] for r in range(49))
                    if abs(s - 1) > 1e-9:
                        valid4 = False
                        break
                if not valid4: break
            if not valid4: break
        
        if not valid4:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
        # Build 4x4x8: split into two 4x4x4 halves, rank 98
        rank = 98
        U8 = np.zeros((16, rank))
        V8 = np.zeros((32, rank))
        W8 = np.zeros((32, rank))
        
        # Part 1: k=0..3 (first 4 columns of B)
        U8[:, 0:49] = U4
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + k, r] = V4[j*4 + k, r]
        W8[:, 0:49] = W4
        
        # Part 2: k=4..7 (second 4 columns of B)
        U8[:, 49:98] = U4
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + 4 + k, 49 + r] = V4[j*4 + k, r]
        for r in range(49):
            for k in range(4):
                for i in range(4):
                    W8[(4+k)*4 + i, 49 + r] = W4[k*4 + i, r]
        
        if time.time() > deadline:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
        return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
        
    except Exception:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
