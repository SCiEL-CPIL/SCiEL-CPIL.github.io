import numpy as np
import time

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, -1]
    ], dtype=float)
    return U, V, W

def _2x2x3_decomp():
    """2x2 x 2x3 multiplication, rank 11 (Strassen 7 + schoolbook 4)"""
    U_str, V_str, W_str = _strassen_2x2x2()
    # Schoolbook for 2x2x1: rank 4
    U_sch = np.zeros((4, 4))
    V_sch = np.zeros((2, 4))
    W_sch = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U_sch[i * 2 + j, r] = 1
            V_sch[j, r] = 1
            W_sch[i, r] = 1
    
    U = np.hstack([U_str, U_sch])  # (4, 11)
    V = np.zeros((6, 11))
    V[0:4, 0:7] = V_str
    V[4:6, 7:11] = V_sch
    W = np.zeros((6, 11))
    W[0:4, 0:7] = W_str
    W[4:6, 7:11] = W_sch
    return U, V, W

def _4x4x4_decomp():
    """4x4 x 4x4 multiplication, rank 49 (Kronecker of Strassen)"""
    U_str, V_str, W_str = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, r] = U_str[i1 * 2 + j1, r1] * U_str[i2 * 2 + j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, r] = V_str[j1 * 2 + k1, r1] * V_str[j2 * 2 + k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, r] = W_str[k1 * 2 + i1, r1] * W_str[k2 * 2 + i2, r2]
    return U, V, W

def _4x4x3_decomp():
    """4x4 x 4x3 multiplication, rank 44 (4 blocks of 2x2x3)"""
    U_223, V_223, W_223 = _2x2x3_decomp()
    U = np.zeros((16, 44))
    V = np.zeros((12, 44))
    W = np.zeros((12, 44))
    
    for b in range(4):
        u_offset = b * 11
        v_offset = b * 11
        w_offset = b * 11
        
        i1 = b // 2
        j1 = b % 2
        
        for i2 in range(2):
            for j2 in range(2):
                i = i1 * 2 + i2
                j = j1 * 2 + j2
                U[i * 4 + j, u_offset:u_offset+11] = U_223[i2 * 2 + j2, :]
        
        for k in range(3):
            for j2 in range(2):
                j = j1 * 2 + j2
                v_row = j * 3 + k
                v_223_row = k * 2 + j2
                V[v_row, v_offset:v_offset+11] = V_223[v_223_row, :]
        
        for k in range(3):
            for i2 in range(2):
                i = i1 * 2 + i2
                w_row = k * 4 + i
                w_223_row = k * 2 + i2
                W[w_row, w_offset:w_offset+11] = W_223[w_223_row, :]
                
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    
    # Decompose 4x4x7 as:
    # C[4x7] = A[4x4] * B[4x7]
    # Split B into B_left[4x4] and B_right[4x3]
    # C = A * B_left + A * B_right
    # Use rank 49 for 4x4x4 and rank 44 for 4x4x3
    # Total rank: 93
    
    U444, V444, W444 = _4x4x4_decomp()  # (16,49), (16,49), (16,49)
    U443, V443, W443 = _4x4x3_decomp()  # (16,44), (12,44), (12,44)
    
    # Extend V444 and W444 to 7 columns
    # V: B is 4x7, indexed by (j,k) -> row j*7+k
    # For 4x4 part: k in 0..3
    V444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V444_full[j * 7 + k, :] = V444[j * 4 + k, :]
    
    W444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W444_full[k * 4 + i, :] = W444[k * 4 + i, :]
    
    # For 4x3 part: k in 4..6
    V443_full = np.zeros((28, 44))
    for j in range(4):
        for k in range(3):
            V443_full[j * 7 + (4 + k), :] = V443[j * 3 + k, :]
    
    W443_full = np.zeros((28, 44))
    for k in range(3):
        for i in range(4):
            W443_full[(4 + k) * 4 + i, :] = W443[k * 4 + i, :]
    
    U = np.hstack([U444, U443])  # (16, 93)
    V = np.hstack([V444_full, V443_full])  # (28, 93)
    W = np.hstack([W444_full, W443_full])  # (28, 93)
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
    
    if time.time() - t0 > 110:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
