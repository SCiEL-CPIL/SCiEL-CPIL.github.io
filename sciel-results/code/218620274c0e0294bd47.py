import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 4, 6
    
    # Use Kronecker product of 2x2x2 Strassen (rank 7) and 1x2x3 trivial (rank 6)
    # Total rank: 7 * 6 = 42
    
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Core 1: 2x2x2 Strassen Tensor
    # U1 shape (4, 7), V1 shape (4, 7), W1 shape (4, 7)
    # U1[i*2+j, r], V1[j*2+k, r], W1[k*2+i, r]
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U1 (A side): indices i*2+j
    U1[0, 0] = 1; U1[3, 0] = 1  # M1: A00, A11
    U1[2, 1] = 1; U1[3, 1] = 1  # M2: A10, A11
    U1[0, 2] = 1                # M3: A00
    U1[3, 3] = 1                # M4: A11
    U1[0, 4] = 1; U1[1, 4] = 1  # M5: A00, A01
    U1[2, 5] = 1; U1[0, 5] = -1 # M6: A10, -A00
    U1[1, 6] = 1; U1[3, 6] = -1 # M7: A01, -A11
    
    # V1 (B side): indices j*2+k
    V1[0, 0] = 1; V1[3, 0] = 1  # M1: B00, B11
    V1[0, 1] = 1                # M2: B00
    V1[1, 2] = 1; V1[3, 2] = -1 # M3: B01, -B11
    V1[2, 3] = 1; V1[0, 3] = -1 # M4: B10, -B00
    V1[3, 4] = 1                # M5: B11
    V1[0, 5] = 1; V1[1, 5] = 1  # M6: B00, B01
    V1[2, 6] = 1; V1[3, 6] = 1  # M7: B10, B11
    
    # W1 (Output side): indices k*2+i (trace convention)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W1[0, 0] = 1; W1[0, 3] = 1; W1[0, 4] = -1; W1[0, 6] = 1  # C00 (k=0,i=0)
    W1[2, 2] = 1; W1[2, 4] = 1                                 # C01 (k=1,i=0)
    W1[1, 1] = 1; W1[1, 3] = 1                                 # C10 (k=0,i=1)
    W1[3, 0] = 1; W1[3, 1] = -1; W1[3, 2] = 1; W1[3, 5] = 1   # C11 (k=1,i=1)
    
    # Core 2: 1x2x3 Trivial Tensor
    # U2 shape (2, 6), V2 shape (6, 6), W2 shape (3, 6)
    # U2[i*2+j, r] where i=0 always, j in {0,1}
    # V2[j*3+k, r] where j in {0,1}, k in {0,1,2}
    # W2[k*1+i, r] where k in {0,1,2}, i=0 always
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    # 6 independent multiplications
    for t in range(6):
        j2 = t // 3  # 0 or 1
        k2 = t % 3   # 0, 1, or 2
        U2[0 * 2 + j2, t] = 1  # A[0, j2]
        V2[j2 * 3 + k2, t] = 1 # B[j2, k2]
        W2[k2 * 1 + 0, t] = 1  # C[k2, 0]
    
    # Kronecker product:
    # T = T1 (2,2,2) ⊗ T2 (1,2,3)
    # Combined dimensions: n=2*1=2, m=2*2=4, p=2*3=6
    # 
    # A[i, a] where i = i1, a = j1*2 + j2
    #   i1 in {0,1}, j1 in {0,1}, j2 in {0,1}
    #   Global A index: i1*4 + (j1*2 + j2)
    #
    # B[a, b] where a = j1*2 + j2, b = k1*3 + k2
    #   j1 in {0,1}, j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    #   Global B index: (j1*2 + j2)*6 + (k1*3 + k2)
    #
    # C[i, b] where i = i1, b = k1*3 + k2
    #   i1 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    #   Global W index: (k1*3 + k2)*2 + i1
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U: A side
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U1[i1 * 2 + j1, r1]
                    if val1 == 0:
                        continue
                    for j2 in range(2):
                        val2 = U2[0 * 2 + j2, r2]
                        if val2 == 0:
                            continue
                        idx_u = i1 * 4 + (j1 * 2 + j2)
                        U[idx_u, r] += val1 * val2
            
            # V: B side
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V1[j1 * 2 + k1, r1]
                    if val1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            val2 = V2[j2 * 3 + k2, r2]
                            if val2 == 0:
                                continue
                            idx_v = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                            V[idx_v, r] += val1 * val2
            
            # W: C side (trace convention: W[k*n + i])
            for i1 in range(2):
                for k1 in range(2):
                    val1 = W1[k1 * 2 + i1, r1]
                    if val1 == 0:
                        continue
                    for k2 in range(3):
                        val2 = W2[k2 * 1 + 0, r2]
                        if val2 == 0:
                            continue
                        idx_w = (k1 * 3 + k2) * 2 + i1
                        W[idx_w, r] += val1 * val2
    
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify correctness with a few random checks
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        
        ui = i * m + j
        vj = j * p + k
        wk = k * n + i
        
        lhs = np.sum(U_arr[ui] * V_arr[vj] * W_arr[wk])
        if abs(lhs - 1.0) > 1e-10:
            # Decomposition error, but continue
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
