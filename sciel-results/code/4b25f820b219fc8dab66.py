"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""
DIMENSIONS = [4, 4, 4]

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
      U[i*2+j, c] = A[i,j] coefficient
      V[j*2+k, c] = B[j,k] coefficient  
      W[k*2+i, c] = C[i,k] coefficient (trace convention: W[k*n+i])
    
    Strassen's 7 products:
      M1 = (a+d)(e+h)
      M2 = (c+d)e
      M3 = a(f-h)
      M4 = d(g-h)
      M5 = (a+b)h
      M6 = (c-d)(g+h)
      M7 = (a-c)(e+f)
    
    Where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[a+h, e+h], [e+f, d+h]] - wait, that's not right.
    C[i,j] = sum_k A[i,k]*B[k,j]
    C[0,0] = a*e + c*g
    C[0,1] = a*f + c*h
    C[1,0] = b*e + d*g
    C[1,1] = b*f + d*h
    
    Using Strassen:
      M1 = (a+d)*(e+h) = a*e + a*h + d*e + d*h
      M2 = (c+d)*e = c*e + d*e
      M3 = a*(f-h) = a*f - a*h
      M4 = d*(g-h) = d*g - d*h
      M5 = (a+b)*h = a*h + b*h
      M6 = (c-d)*(g+h) = c*g + c*h - d*g - d*h
      M7 = (a-c)*(e+f) = a*e + a*f - c*e - c*f
    
    Then:
      C[0,0] = M1 - M4 - M5 + M7
      C[0,1] = M3 + M5
      C[1,0] = M2 + M4
      C[1,1] = M1 - M2 + M6 + M7
    """
    n = 2
    rank = 7
    U = [[0] * rank for _ in range(n * n)]  # 4x7
    V = [[0] * rank for _ in range(n * n)]  # 4x7
    W = [[0] * rank for _ in range(n * n)]  # 4x7
    
    # U rows: A[i,j] indexed as i*n+j -> a=U[0], b=U[1], c=U[2], d=U[3]
    # V rows: B[j,k] indexed as j*n+k -> e=V[0], f=V[1], g=V[2], h=V[3]
    # W rows: C indexed by k*n+i -> C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    
    # M1 = (a+d)*(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    # C[0,0] += M1, C[1,1] += M1
    W[0][0] = 1   # C[0,0]
    W[3][0] = 1   # C[1,1]
    
    # M2 = (c+d)*e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    # C[1,0] += M2, C[1,1] -= M2
    W[1][1] = 1   # C[1,0]
    W[3][1] = -1  # C[1,1]
    
    # M3 = a*(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # -h
    # C[0,1] += M3
    W[2][2] = 1   # C[0,1]
    
    # M4 = d*(g-h)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[3][3] = -1 # -h
    # C[1,0] += M4, C[0,0] -= M4
    W[1][3] = 1   # C[1,0]
    W[0][3] = -1  # C[0,0]
    
    # M5 = (a+b)*h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    # C[0,1] += M5, C[0,0] -= M5
    W[2][4] = 1   # C[0,1]
    W[0][4] = -1  # C[0,0]
    
    # M6 = (c-d)*(g+h)
    U[2][5] = 1  # c
    U[3][5] = -1 # -d
    V[2][5] = 1  # g
    V[3][5] = 1  # h
    # C[1,1] += M6
    W[3][5] = 1   # C[1,1]
    
    # M7 = (a-c)*(e+f)
    U[0][6] = 1  # a
    U[2][6] = -1 # -c
    V[0][6] = 1  # e
    V[1][6] = 1  # f
    # C[0,0] += M7, C[1,1] += M7
    W[0][6] = 1   # C[0,0]
    W[3][6] = 1   # C[1,1]
    
    return U, V, W

def _kron_decompose(U1, V1, W1, n1):
    """
    Decompose tensor product T2 = T1 x T1 using Kronecker product structure.
    If T1 has rank r with factors (U1, V1, W1) for size n1,
    then T2 = T1 x T1 has rank r^2 for size n1^2.
    
    For 2x2 -> 4x4:
      U4[I*4+J, (r1*7+r2)] = U1[i1*2+j1, r1] * U1[i2*2+j2, r2]
      where I = i1*2+i2, J = j1*2+j2
      V4[J*4+K, (r1*7+r2)] = V1[j1*2+k1, r1] * V1[j2*2+k2, r2]
      where J = j1*2+j2, K = k1*2+k2
      W4[K*4+I, (r1*7+r2)] = W1[k1*2+i1, r1] * W1[k2*2+i2, r2]
      where K = k1*2+k2, I = i1*2+i2
    """
    r1 = len(U1[0])  # rank of 2x2 (should be 7)
    n2 = n1 * n1  # 4
    rank = r1 * r1  # 49
    
    U2 = [[0] * rank for _ in range(n2 * n2)]  # 16x49
    V2 = [[0] * rank for _ in range(n2 * n2)]  # 16x49
    W2 = [[0] * rank for _ in range(n2 * n2)]  # 16x49
    
    for i1 in range(n1):
        for i2 in range(n1):
            I = i1 * n1 + i2
            for j1 in range(n1):
                for j2 in range(n1):
                    J = j1 * n1 + j2
                    for k1 in range(n1):
                        for k2 in range(n1):
                            K = k1 * n1 + k2
                            # For each pair of Strassen terms
                            for r1_idx in range(r1):
                                for r2_idx in range(r1):
                                    col = r1_idx * r1 + r2_idx
                                    u1_a = U1[i1 * n1 + j1][r1_idx]
                                    u1_b = U1[i2 * n1 + j2][r2_idx]
                                    if u1_a != 0 and u1_b != 0:
                                        U2[I * n2 + J][col] = u1_a * u1_b
                                    
                                    v1_a = V1[j1 * n1 + k1][r1_idx]
                                    v1_b = V1[j2 * n1 + k2][r2_idx]
                                    if v1_a != 0 and v1_b != 0:
                                        V2[J * n2 + K][col] = v1_a * v1_b
                                    
                                    w1_a = W1[k1 * n1 + i1][r1_idx]
                                    w1_b = W1[k2 * n1 + i2][r2_idx]
                                    if w1_a != 0 and w1_b != 0:
                                        W2[K * n2 + I][col] = w1_a * w1_b
    
    return U2, V2, W2

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-49 decomposition for 4x4 matrix multiplication
    n, m, p = DIMENSIONS
    
    # Get 2x2 Strassen decomposition
    U1, V1, W1 = _strassen_2x2()
    
    # Build 4x4 via Kronecker product
    U, V, W = _kron_decompose(U1, V1, W1, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
