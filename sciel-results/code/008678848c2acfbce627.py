import numpy as np

DIMENSIONS = [2, 3, 3]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1) > 1e-9:
                    return False, (i, j, k, val)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                val = np.sum(U[a] * V[b] * W[c])
                if valid:
                    if abs(val - 1) > 1e-9:
                        return False, (a, b, c, val)
                else:
                    if abs(val) > 1e-9:
                        return False, (a, b, c, val)
    return True, None

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
        
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # W indices: 0,1,2,3
    
    # M1 = a11(b22 - b12)
    add_term(0, 1, 4, 1, 2, 1, 0) # a11, b22 -> c12
    add_term(0, 0, 1, -1, 2, 1, 0) # a11, b12 -> c12
    add_term(0, 0, 4, 1, 3, 1, 0) # a11, b22 -> c22
    add_term(0, 0, 1, -1, 3, 1, 0) # a11, b12 -> c22
    
    # M2 = (a11 + a12)b22
    add_term(0, 1, 4, 1, 0, -1, 1) # a11, b22 -> c11
    add_term(1, 1, 4, 1, 0, -1, 1) # a12, b22 -> c11
    add_term(0, 1, 4, 1, 2, 1, 1) # a11, b22 -> c12
    add_term(1, 1, 4, 1, 2, 1, 1) # a12, b22 -> c12
    
    # M3 = (a21 + a22)b11
    add_term(3, 1, 0, 1, 1, 1, 2) # a21, b11 -> c21
    add_term(4, 1, 0, 1, 1, 1, 2) # a22, b11 -> c21
    add_term(3, 1, 0, 1, 3, -1, 2) # a21, b11 -> c22
    add_term(4, 1, 0, 1, 3, -1, 2) # a22, b11 -> c22
    
    # M4 = a22(b11 - b21)
    add_term(4, 1, 0, 1, 0, 1, 3) # a22, b11 -> c11
    add_term(4, 1, 3, -1, 0, 1, 3) # a22, b21 -> c11
    add_term(4, 1, 0, 1, 1, 1, 3) # a22, b11 -> c21
    add_term(4, 1, 3, -1, 1, 1, 3) # a22, b21 -> c21
    
    # M5 = (a11 + a22)(b11 + b22)
    add_term(0, 1, 0, 1, 0, 1, 4) # a11, b11 -> c11
    add_term(0, 1, 4, 1, 0, 1, 4) # a11, b22 -> c11
    add_term(4, 1, 0, 1, 0, 1, 4) # a22, b11 -> c11
    add_term(4, 1, 4, 1, 0, 1, 4) # a22, b22 -> c11
    add_term(0, 1, 0, 1, 3, 1, 4) # a11, b11 -> c22
    add_term(0, 1, 4, 1, 3, 1, 4) # a11, b22 -> c22
    add_term(4, 1, 0, 1, 3, 1, 4) # a22, b11 -> c22
    add_term(4, 1, 4, 1, 3, 1, 4) # a22, b22 -> c22
    
    # M6 = (a12 - a22)(b21 + b22)
    add_term(1, 1, 3, 1, 0, 1, 5) # a12, b21 -> c11
    add_term(1, 1, 4, 1, 0, 1, 5) # a12, b22 -> c11
    add_term(4, -1, 3, 1, 0, 1, 5) # a22, b21 -> c11
    add_term(4, -1, 4, 1, 0, 1, 5) # a22, b22 -> c11
    
    # M7 = (a11 - a21)(b11 + b12)
    add_term(0, 1, 0, 1, 3, 1, 6) # a11, b11 -> c22
    add_term(0, 1, 1, 1, 3, 1, 6) # a11, b12 -> c22
    add_term(3, -1, 0, 1, 3, 1, 6) # a21, b11 -> c22
    add_term(3, -1, 1, 1, 3, 1, 6) # a21, b12 -> c22

    # A[0:2, 0:2] * B[0:2, 2] (col 2 of B, which is index 2 and 5)
    # B indices: 2 (b13), 5 (b23)
    # W indices: 2 (c12), 3 (c22)
    
    # a11 b13 -> c12
    add_term(0, 1, 2, 1, 2, 1, 7)
    # a12 b23 -> c12
    add_term(1, 1, 5, 1, 2, 1, 8)
    # a21 b13 -> c22
    add_term(3, 1, 2, 1, 3, 1, 9)
    # a22 b23 -> c22
    add_term(4, 1, 5, 1, 3, 1, 10)

    # A[0:2, 2] * B[2, 0:3]
    # A indices: 2 (a13), 5 (a23)
    # B indices: 6 (b31), 7 (b32), 8 (b33)
    # W indices: 0,1,2 (c10,c11,c12) and 3,4,5 (c20,c21,c22)
    
    # a13 b31 -> c10
    add_term(2, 1, 6, 1, 0, 1, 11)
    # a13 b32 -> c11
    add_term(2, 1, 7, 1, 1, 1, 12)
    # a13 b33 -> c12
    add_term(2, 1, 8, 1, 2, 1, 13)
    # a23 b31 -> c20
    add_term(5, 1, 6, 1, 3, 1, 14)
    # a23 b32 -> c21
    add_term(5, 1, 7, 1, 4, 1, 15)
    # a23 b33 -> c22
    add_term(5, 1, 8, 1, 5, 1, 16)

    return {"U": U, "V": V, "W": W}
