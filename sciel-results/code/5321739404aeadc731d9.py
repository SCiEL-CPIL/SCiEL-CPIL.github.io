"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    # 7 terms: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # 6 terms: A[0:2, 2] * B[2, 0:3] (outer product)
    # 4 terms: A[0:2, 0:2] * B[0:2, 2] (2x2 * 2x1)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A11=a, A12=b, A21=d, A22=e
    # B11=g, B12=h, B21=j, B22=k
    # C11 = M1 - M2 + M3 + M7
    # C12 = M4 + M5
    # C21 = M2 + M6
    # C22 = M1 - M5 + M6 + M7
    
    # M1 = (a+d)(g+k) -> U: a,d; V: g,k; W: C11, C22
    # M2 = (a+d)j -> U: a,d; V: j; W: C11, C21
    # M3 = a(g-j) -> U: a; V: g,j; W: C11
    # M4 = e(k-h) -> U: e; V: k,h; W: C12
    # M5 = (b+e)h -> U: b,e; V: h; W: C12
    # M6 = (d-b)(g+j) -> U: d,b; V: g,j; W: C21
    # M7 = (a-e)(k+h) -> U: a,e; V: k,h; W: C11, C22
    
    col = 0
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    # M2
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    # M3
    U[0][col] = 1
    V[0][col] = 1; V[1][col] = -1
    W[0][col] = 1
    col += 1
    # M4
    U[1][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[1][col] = 1
    col += 1
    # M5
    U[1][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1
    col += 1
    # M6
    U[3][col] = 1; U[1][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[1][col] = 1
    col += 1
    # M7
    U[0][col] = 1; U[4][col] = -1
    V[1][col] = 1; V[0][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # 2. A[0:2, 2] * B[2, 0:3]
    # A: c (idx 2), f (idx 5)
    # B: m (idx 6), n (idx 7), o (idx 8)
    # C: c*m -> C00, c*n -> C01, c*o -> C02
    #     f*m -> C10, f*n -> C11, f*o -> C12
    
    # c*m
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    # c*n
    U[2][col] = 1
    V[7][col] = 1
    W[1][col] = 1
    col += 1
    # c*o
    U[2][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    # f*m
    U[5][col] = 1
    V[6][col] = 1
    W[3][col] = 1
    col += 1
    # f*n
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    # f*o
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1
    col += 1
    
    # 3. A[0:2, 0:2] * B[0:2, 2]
    # A: a(0), b(1), d(3), e(4)
    # B: i(2), l(5)
    # C: a*i -> C02, b*l -> C02, d*i -> C12, e*l -> C12
    
    # a*i
    U[0][col] = 1
    V[2][col] = 1
    W[2][col] = 1
    col += 1
    # b*l
    U[1][col] = 1
    V[5][col] = 1
    W[2][col] = 1
    col += 1
    # d*i
    U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1
    col += 1
    # e*l
    U[4][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
