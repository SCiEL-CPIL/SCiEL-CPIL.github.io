import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Attempt rank 30: 3 Strassen blocks (rank 7 each) + custom 2x2x3 (rank 9)
    # Strassen for (0,1), (2,3), (4,5); custom for (6,7,8)
    
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 30
    V = [[0] * rank for _ in range(m * p)]  # 18 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1 = (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[9+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        # M2 = (c+d)e
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        # M3 = a(f-h)
        U[0][col] = 1
        V[k1][col] = 1; V[9+k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        # M4 = d(g-e)
        U[3][col] = 1
        V[9+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1
        # M5 = (a+b)h
        U[0][col] = 1; U[1][col] = 1
        V[9+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        col += 1
        # M6 = (c-a)(e+f)
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        # M7 = (b-d)(g+h)
        U[1][col] = 1; U[3][col] = -1
        V[9+k0][col] = 1; V[9+k1][col] = 1
        W[2*k0][col] = 1
        col += 1
    
    # Add 3 Strassen blocks
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    
    # Now custom rank-9 decomposition for 2x2x3 (columns 6,7,8)
    # k0=6, k1=7, k2=8
    k0, k1, k2 = 6, 7, 8
    
    # Term 1: (a+d)(e0+e1) -> C[0,k0]+1, C[1,k1]+1
    U[0][col] = 1; U[3][col] = 1
    V[k0][col] = 1; V[k1][col] = 1
    W[2*k0][col] = 1; W[2*k1+1][col] = 1
    col += 1
    
    # Term 2: (c+d)(e0+e1) -> C[1,k0]+1, C[1,k1]-1
    U[2][col] = 1; U[3][col] = 1
    V[k0][col] = 1; V[k1][col] = 1
    W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
    col += 1
    
    # Term 3: a(f0-f1) -> C[0,k1]+1, C[1,k1]+1
    U[0][col] = 1
    V[k1][col] = 1; V[9+k1][col] = -1
    W[2*k1][col] = 1; W[2*k1+1][col] = 1
    col += 1
    
    # Term 4: d(g0-g1) -> C[0,k0]+1, C[1,k0]+1
    U[3][col] = 1
    V[9+k0][col] = 1; V[k0][col] = -1
    W[2*k0][col] = 1; W[2*k0+1][col] = 1
    col += 1
    
    # Term 5: (a+b)(h0+h1) -> C[0,k0]-1, C[0,k1]+1
    U[0][col] = 1; U[1][col] = 1
    V[9+k0][col] = 1; V[9+k1][col] = 1
    W[2*k0][col] = -1; W[2*k1][col] = 1
    col += 1
    
    # Term 6: (c-a)(g0+g1) -> C[1,k1]+1
    U[2][col] = 1; U[0][col] = -1
    V[9+k0][col] = 1; V[9+k1][col] = 1
    W[2*k1+1][col] = 1
    col += 1
    
    # Term 7: (b-d)(f0+f1) -> C[0,k0]+1
    U[1][col] = 1; U[3][col] = -1
    V[k0][col] = 1; V[k1][col] = 1
    W[2*k0][col] = 1
    col += 1
    
    # Term 8: a*e2 -> C[0,k2]+1
    U[0][col] = 1
    V[k2][col] = 1
    W[2*k2][col] = 1
    col += 1
    
    # Term 9: d*g2 -> C[1,k2]+1
    U[3][col] = 1
    V[9+k2][col] = 1
    W[2*k2+1][col] = 1
    col += 1
    
    # Verify: we need C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2]
    # and C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2]
    # But terms 8 and 9 only give A[0,0]*B[0,k2] and A[1,1]*B[1,k2]
    # We're missing A[0,1]*B[1,k2] and A[1,0]*B[0,k2]
    # This decomposition is WRONG for k2. Let me fix this.
    
    # Actually, this approach doesn't work cleanly. Let me fall back to rank 32
    # but try a slightly different structure.
    
    # Reset and use the proven rank-32 approach
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_safe(k0, k1):
        nonlocal col
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[9+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        U[0][col] = 1
        V[k1][col] = 1; V[9+k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        U[3][col] = 1
        V[9+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1
        U[0][col] = 1; U[1][col] = 1
        V[9+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        col += 1
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        U[1][col] = 1; U[3][col] = -1
        V[9+k0][col] = 1; V[9+k1][col] = 1
        W[2*k0][col] = 1
        col += 1
    
    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
        U[1][col] = 1; V[9+k][col] = 1; W[2*k][col] = 1; col += 1
        U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
        U[3][col] = 1; V[9+k][col] = 1; W[2*k+1][col] = 1; col += 1
    
    add_strassen_safe(0, 1)
    add_strassen_safe(2, 3)
    add_strassen_safe(4, 5)
    add_strassen_safe(6, 7)
    add_trivial(8)
    
    return {"U": U, "V": V, "W": W}
