import numpy as np

def construct():
    n, m, p = 3, 3, 3
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Strassen-based decomposition
    # Partition into 2x2 and 1x1 blocks
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2], A3 = A[2, 0:2], A4 = A[2, 2]
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2], B3 = B[2, 0:2], B4 = B[2, 2]
    # C1 = C[0:2, 0:2], C2 = C[0:2, 2], C3 = C[2, 0:2], C4 = C[2, 2]
    
    # C1 = A1 * B1 (use Strassen, 7 terms)
    # C2 = A1 * B2 + A2 * B4 (use schoolbook, 6 terms)
    # C3 = A3 * B1 + A4 * B3 (use schoolbook, 6 terms)
    # C4 = A3 * B2 + A4 * B4 (use schoolbook, 3 terms)
    # Total: 7 + 6 + 6 + 3 = 22 terms
    
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    
    # Helper function to add a schoolbook term
    def add_schoolbook_term(i, j, k):
        nonlocal r
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        r += 1
    
    # Strassen terms for C1 = A1 * B1
    # A1 = [[a00, a01], [a10, a11]]
    # B1 = [[b00, b01], [b10, b11]]
    # C1 = [[c00, c01], [c10, c11]]
    
    # Index mapping:
    # a00 = A[0,0] -> U[0, r]
    # a01 = A[0,1] -> U[1, r]
    # a10 = A[1,0] -> U[3, r]
    # a11 = A[1,1] -> U[4, r]
    
    # b00 = B[0,0] -> V[0, r]
    # b01 = B[0,1] -> V[1, r]
    # b10 = B[1,0] -> V[3, r]
    # b11 = B[1,1] -> V[4, r]
    
    # c00 = C[0,0] -> W[0, r]
    # c01 = C[0,1] -> W[3, r]
    # c10 = C[1,0] -> W[1, r]
    # c11 = C[1,1] -> W[4, r]
    
    # Strassen:
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, r] = 1; U[4, r] = 1
    V[0, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[3, r] = 1; W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M2 = (a10 + a11) * b00
    U[3, r] = 1; U[4, r] = 1
    V[0, r] = 1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M3 = a00 * (b01 - b11)
    U[0, r] = 1
    V[1, r] = 1; V[4, r] = -1
    W[3, r] = 1
    r += 1
    
    # M4 = a11 * (b10 - b00)
    U[4, r] = 1
    V[3, r] = 1; V[0, r] = -1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M5 = (a00 + a01) * b11
    U[0, r] = 1; U[1, r] = 1
    V[4, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # M6 = (a10 - a00) * (b00 + b01)
    U[3, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[1, r] = 1; W[4, r] = 1
    r += 1
    
    # M7 = (a01 - a11) * (b10 + b11)
    U[1, r] = 1; U[4, r] = -1
    V[3, r] = 1; V[4, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # Schoolbook terms for C2 = A1 * B2 + A2 * B4
    # C2 = C[0:2, 2] = [[c02], [c12]]
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # b02 = B[0,2] -> V[2, r]
    # b12 = B[1,2] -> V[5, r]
    # b22 = B[2,2] -> V[8, r]
    
    # a02 = A[0,2] -> U[2, r]
    # a12 = A[1,2] -> U[5, r]
    
    # c02 = C[0,2] -> W[2*3+0, r] = W[6, r]
    # c12 = C[1,2] -> W[2*3+1, r] = W[7, r]
    
    add_schoolbook_term(0, 0, 2)  # a00*b02 -> c02
    add_schoolbook_term(0, 1, 2)  # a01*b12 -> c02
    add_schoolbook_term(0, 2, 2)  # a02*b22 -> c02
    add_schoolbook_term(1, 0, 2)  # a10*b02 -> c12
    add_schoolbook_term(1, 1, 2)  # a11*b12 -> c12
    add_schoolbook_term(1, 2, 2)  # a12*b22 -> c12
    
    # Schoolbook terms for C3 = A3 * B1 + A4 * B3
    # C3 = C[2, 0:2] = [[c20, c21]]
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    
    # a20 = A[2,0] -> U[6, r]
    # a21 = A[2,1] -> U[7, r]
    # a22 = A[2,2] -> U[8, r]
    
    # b20 = B[2,0] -> V[6, r]
    # b21 = B[2,1] -> V[7, r]
    
    # c20 = C[2,0] -> W[0*3+2, r] = W[2, r]
    # c21 = C[2,1] -> W[1*3+2, r] = W[5, r]
    
    add_schoolbook_term(2, 0, 0)  # a20*b00 -> c20
    add_schoolbook_term(2, 1, 0)  # a21*b10 -> c20
    add_schoolbook_term(2, 2, 0)  # a22*b20 -> c20
    add_schoolbook_term(2, 0, 1)  # a20*b01 -> c21
    add_schoolbook_term(2, 1, 1)  # a21*b11 -> c21
    add_schoolbook_term(2, 2, 1)  # a22*b21 -> c21
    
    # Schoolbook terms for C4 = A3 * B2 + A4 * B4
    # C4 = C[2, 2] = [[c22]]
    # c22 = a20*b02 + a21*b12 + a22*b22
    
    add_schoolbook_term(2, 0, 2)  # a20*b02 -> c22
    add_schoolbook_term(2, 1, 2)  # a21*b12 -> c22
    add_schoolbook_term(2, 2, 2)  # a22*b22 -> c22
    
    # Verify
    U_np = U
    V_np = V
    W_np = W
    
    # Check: sum_r U[a,r] * V[b,r] * W[c,r] == T[a,b,c]
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = sum(U_np[a, r] * V_np[b, r] * W_np[c, r] for r in range(rank))
                if abs(val - T[a, b, c]) > 1e-9:
                    print(f"Error at ({a},{b},{c}): expected {T[a,b,c]}, got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
