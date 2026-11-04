import numpy as np
import time

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, -1]
    ])
    return U, V, W

def _2x2x7_decomp():
    """Decompose 2x2x7 matrix multiplication: C[2x7] = A[2x2] * B[2x7]
    Split B into B1[2x2] and B2[2x5].
    Use Strassen (rank 7) for A*B1, schoolbook (rank 20) for A*B2.
    Total rank: 27.
    """
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # Schoolbook for 2x2x5: rank = 2*2*5 = 20
    U_sch = np.zeros((4, 20))
    V_sch = np.zeros((10, 20))  # B is 2x5, so 10 elements
    W_sch = np.zeros((10, 20))  # C is 2x5, so 10 elements
    for i in range(2):
        for j in range(2):
            for k in range(5):
                r = i * 2 * 5 + j * 5 + k
                U_sch[i * 2 + j, r] = 1
                V_sch[j * 5 + k, r] = 1
                W_sch[k * 2 + i, r] = 1
    
    U = np.hstack([U_str, U_sch])  # (4, 27)
    
    V = np.zeros((14, 27))  # B is 2x7, so 14 elements
    # Strassen part: B[0:2, 0:2] -> rows 0:4
    V[0:4, 0:7] = V_str
    # Schoolbook part: B[0:2, 2:7] -> rows 4:14
    V[4:14, 7:27] = V_sch
    
    W = np.zeros((14, 27))  # C is 2x7, so 14 elements
    # Strassen part: C[0:2, 0:2] -> rows 0:4
    W[0:4, 0:7] = W_str
    # Schoolbook part: C[0:2, 2:7] -> rows 4:14
    W[4:14, 7:27] = W_sch
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    
    # Decompose 4x4x7 as 4 independent 2x2x7 sub-problems:
    # C[0:2, :] = A[0:2, 0:2]*B[0:2, :] + A[0:2, 2:4]*B[2:4, :]
    # C[2:4, :] = A[2:4, 0:2]*B[0:2, :] + A[2:4, 2:4]*B[2:4, :]
    # Each term is a 2x2x7 product, rank 27 each, total rank 108.
    
    # But 108 > 93, so this is worse. Let me use the existing approach.
    
    U444, V444, W444 = _4x4x4_decomp()
    U443, V443, W443 = _4x4x3_decomp()
    
    V444_full = np.zeros((28, 49))
    W444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V444_full[j * 7 + k, :] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W444_full[k * 4 + i, :] = W444[k * 4 + i, :]
            
    V443_full = np.zeros((28, 44))
    W443_full = np.zeros((28, 44))
    for j in range(4):
        for k in range(3):
            V443_full[j * 7 + (4 + k), :] = V443[j * 3 + k, :]
    for k in range(3):
        for i in range(4):
            W443_full[(4 + k) * 4 + i, :] = W443[k * 4 + i, :]
            
    U = np.hstack([U444, U443])
    V = np.hstack([V444_full, V443_full])
    W = np.hstack([W444_full, W443_full])
    
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
                    
    if time.time() - t0 > 120:
        # Emergency fallback
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

def _schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U[i * 2 + j, r] = 1
            V[j, r] = 1
            W[i, r] = 1
    return U, V, W

def _2x2x3_decomp():
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sch, V_sch, W_sch = _schoolbook_2x2x1()
    
    U = np.hstack([U_str, U_sch])
    
    V = np.zeros((6, 11))
    V[0:4, 0:7] = V_str
    V[4:6, 7:11] = V_sch
    
    W = np.zeros((6, 11))
    W[0:4, 0:7] = W_str
    W[4:6, 7:11] = W_sch
    
    return U, V, W

def _4x4x3_decomp():
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
        j_b = j1
        i_c = i1
        
        for i2 in range(2):
            for j2 in range(2):
                i = i1 * 2 + i2
                j = j1 * 2 + j2
                U[i * 4 + j, u_offset:u_offset+11] = U_223[i2 * 2 + j2, :]
                
        for k in range(3):
            for j2 in range(2):
                j = j_b * 2 + j2
                v_row = j * 3 + k
                v_223_row = k * 2 + j2
                V[v_row, v_offset:v_offset+11] = V_223[v_223_row, :]
                
        for k in range(3):
            for i2 in range(2):
                i = i_c * 2 + i2
                w_row = k * 4 + i
                w_223_row = k * 2 + i2
                W[w_row, w_offset:w_offset+11] = W_223[w_223_row, :]
                
    return U, V, W

def _4x4x4_decomp():
    U_str, V_str, W_str = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    
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
                            
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, r] = V_str[j1 * 2 + k1, r1] * V_str[j2 * 2 + k2, r2]
                            
                            for i1 in range(2):
                                for i2 in range(2):
                                    i = i1 * 2 + i2
                                    W[k * 4 + i, r] = W_str[k1 * 2 + i1, r1] * W_str[k2 * 2 + i2, r2]
                                    
    return U, V, W
