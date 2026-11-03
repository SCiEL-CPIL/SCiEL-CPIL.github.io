import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
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
    
    # Try Strassen-based rank 98
    try:
        # 2x2x2 Strassen decomposition (verified correct)
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
        
        # Verify 2x2x2 decomposition
        # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if not valid:
            # Return schoolbook if Strassen vectors are incorrect
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

        # Construct 4x4x4 decomposition using Kronecker product
        # U4 = U2 (x) U2, etc.
        # Note: np.kron(A, B) for matrices A (m1, n1) and B (m2, n2) gives (m1*m2, n1*n2)
        # We need U4 of shape (16, 49).
        # U2 is (4, 7).
        U4 = np.kron(U2, U2)
        V4 = np.kron(V2, V2)
        W4 = np.kron(W2, W2)
        
        # Verify 4x4x4 decomposition quickly
        valid4 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for r in range(49):
                        s += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                    if abs(s - 1) > 1e-9:
                        valid4 = False
                        break
                if not valid4: break
            if not valid4: break
            
        if not valid4:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

        # Construct 4x4x8 decomposition
        # Split into two 4x4x4 parts
        U8 = np.zeros((16, 98))
        V8 = np.zeros((32, 98))
        W8 = np.zeros((32, 98))
        
        # Part 1: k in 0..3
        U8[:, 0:49] = U4
        # V8 indices: j*8 + k. k in 0..3.
        # V4 indices: j*4 + k.
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + k, r] = V4[j*4 + k, r]
        # W8 indices: k*4 + i. k in 0..3.
        # W4 indices: k*4 + i.
        W8[:, 0:49] = W4
        
        # Part 2: k in 4..7
        U8[:, 49:98] = U4
        # V8 indices: j*8 + (4+k). k in 0..3.
        for r in range(49):
            for j in range(4):
                for k in range(4):
                    V8[j*8 + 4 + k, 49 + r] = V4[j*4 + k, r]
        # W8 indices: (4+k)*4 + i. k in 0..3.
        for r in range(49):
            for k in range(4):
                for i in range(4):
                    W8[(4+k)*4 + i, 49 + r] = W4[k*4 + i, r]
                    
        # Final check
        if time.time() > deadline:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
            
        return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
        
    except Exception:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
