import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    Uses the rank-7 Strassen decomposition for 2x2x2, then takes the Kronecker
    product to get a rank-49 decomposition for 4x4x4.
    
    Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    The decomposition satisfies: for all i,j,k,
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    def _schoolbook():
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Correct Strassen's 2x2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # U2: rows indexed by i*2+j (i=row of A, j=col of A)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A11: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, -1],  # A12: M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # A21: M2, M6
        [1, 1, 0, 1, 0, 0, -1]   # A22: M1, M2, M4, M7
    ], dtype=int)
    
    # V2: rows indexed by j*2+k (j=row of B, k=col of B)
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # B11: M1, M2, M6
        [0, 0, 1, 0, 0, 1, 0],  # B12: M3, M6
        [0, 0, 0, 1, 0, 0, 1],  # B21: M4, M7
        [1, 0, -1, 0, 1, 0, 1]  # B22: M1, M3, M5, M7
    ], dtype=int)
    
    # W2: rows indexed by k*2+i (k=row of C, i=col of C)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C11: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2, M4
        [1, -1, 1, 0, 0, 1, 0]   # C22: M1, M2, M3, M6
    ], dtype=int)
    
    # Verify 2x2 Strassen correctness
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i]))
                if s != 1:
                    return _schoolbook()
    
    # Kronecker products give 16x49 matrices
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Compute row permutations to match task's linear indexing
    # U: target row S = i*4+j, where i = i1*2+i2, j = j1*2+j2
    #   Kron row R = (i1*2+j1)*4 + (i2*2+j2)
    # V: target row S = j*4+k, where j = j1*2+j2, k = k1*2+k2
    #   Kron row R = (j1*2+k1)*4 + (j2*2+k2)
    # W: target row S = k*4+i, where k = k1*2+k2, i = i1*2+i2
    #   Kron row R = (k1*2+i1)*4 + (k2*2+i2)
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for S in range(16):
        # U permutation
        i = S // 4
        j = S % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        perm_U[S] = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
        
        # V permutation
        j = S // 4
        k = S % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        perm_V[S] = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
        
        # W permutation
        k = S // 4
        i = S % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        perm_W[S] = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify 4x4 correctness
    valid4 = True
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = int(np.dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i]))
                    if s != 1:
                        valid4 = False
                        break
                if not valid4:
                    break
            if not valid4:
                break
    
    if not valid4:
        return _schoolbook()
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
