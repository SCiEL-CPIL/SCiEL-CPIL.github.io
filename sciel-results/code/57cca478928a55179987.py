import numpy as np
import time

def _strassen_2x2():
    # Returns U, V, W for 2x2 Strassen (Rank 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1
    
    return U, V, W

def _coppersmith_winz_2x2():
    # Returns U, V, W for 2x2 Coppersmith-Winograd (Rank 6)
    # This is a known rank-6 decomposition for 2x2 matrix multiplication.
    # The entries are integers.
    U = np.zeros((4, 6))
    V = np.zeros((4, 6))
    W = np.zeros((4, 6))
    
    # The decomposition is given by the following 6 products:
    # M1 = (A00 + A01 + A10 - A11)(B00 + B10 - B01 - B11)
    # M2 = (A00 - A11)(B00 - B11)
    # M3 = (A01 - A10)(B01 - B10)
    # M4 = (A00 + A01 + A10 - A11)(B00 - B10 + B01 + B11)
    # M5 = (A00 - A11)(B00 + B11)
    # M6 = (A01 - A10)(B01 + B10)
    
    # Wait, this doesn't look right. Let me use the standard rank-6 decomposition.
    # The standard rank-6 decomposition for 2x2 matrix multiplication is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A00)(B00 + B10)
    # This is rank 6, but Strassen is rank 7. Wait, Strassen is rank 7.
    # The rank of 2x2 matrix multiplication is 7. So rank 6 is impossible.
    # I made a mistake. The rank of 2x2 matrix multiplication is 7.
    # So I cannot use rank 6. I will use Strassen rank 7.
    
    # Let me return the Strassen decomposition instead.
    return _strassen_2x2()

def _coppersmith_winz_3x3():
    # Returns U, V, W for 3x3 Coppersmith-Winograd (Rank 23)
    # The rank of 3x3 matrix multiplication is 23.
    # This is a known decomposition.
    # I will use the schoolbook rank 27 for now, as finding the exact rank-23
    # decomposition is complex.
    return _schoolbook_3x3()

def _schoolbook_3x3():
    # Returns U, V, W for 3x3 Schoolbook (Rank 27)
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, r] = 1
                V[j*3 + k, r] = 1
                W[k*3 + i, r] = 1
                r += 1
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    # Compute the tensor product of two decompositions
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 * R2
    
    n1 = U1.shape[0]
    m1 = V1.shape[0]
    p1 = W1.shape[0]
    n2 = U2.shape[0]
    m2 = V2.shape[0]
    p2 = W2.shape[0]
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n, R))
    V = np.zeros((m, R))
    W = np.zeros((p, R))
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            for i1 in range(n1):
                for i2 in range(n2):
                    idx = i1 * n2 + i2
                    U[idx, r] = U1[i1, r1] * U2[i2, r2]
            for j1 in range(m1):
                for j2 in range(m2):
                    idx = j1 * m2 + j2
                    V[idx, r] = V1[j1, r1] * V2[j2, r2]
            for k1 in range(p1):
                for k2 in range(p2):
                    idx = k1 * p2 + k2
                    W[idx, r] = W1[k1, r1] * W2[k2, r2]
                    
    return U, V, W

def _verify(n, m, p, U, V, W, tol=1e-8):
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check some zero entries
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1 = a // m; j1 = a % m
        j2 = b // p; k1 = b % p
        k2 = c // n; i2 = c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Build 189 rank decomposition using tensor product of 3x3 schoolbook and 2x2 Strassen
    U3, V3, W3 = _schoolbook_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    U, V, W = _tensor_product(U3, V3, W3, U2, V2, W2)
    
    # Verify
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 216
        U_sb = np.zeros((36, 216))
        V_sb = np.zeros((36, 216))
        W_sb = np.zeros((36, 216))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_sb[i*6+j, col] = 1
                    V_sb[j*6+k, col] = 1
                    W_sb[k*6+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
