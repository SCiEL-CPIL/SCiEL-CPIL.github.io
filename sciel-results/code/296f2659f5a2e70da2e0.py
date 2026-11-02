import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = 2, 2, 15
    
    # Strategy: Block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # Indices:
    # U: 0=a, 1=b, 2=c, 3=d
    # V: j*p + k
    # W: k*n + i
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1 = (a+d)(e+h)
        U[0][r1] = 1
        U[3][r1] = 1
        V[0*p + k0][r1] = 1
        V[1*p + k1][r1] = 1
        
        # P2 = (c+d)e
        U[2][r2] = 1
        U[3][r2] = 1
        V[0*p + k0][r2] = 1
        
        # P3 = a(f-h)
        U[0][r3] = 1
        V[0*p + k1][r3] = 1
        V[1*p + k1][r3] = -1
        
        # P4 = d(g-e)
        U[3][r4] = 1
        V[1*p + k0][r4] = 1
        V[0*p + k0][r4] = -1
        
        # P5 = (a+b)h
        U[0][r5] = 1
        U[1][r5] = 1
        V[1*p + k1][r5] = 1
        
        # P6 = (c-a)(e+f)
        U[2][r6] = 1
        U[0][r6] = -1
        V[0*p + k0][r6] = 1
        V[0*p + k1][r6] = 1
        
        # P7 = (b-d)(g+h)
        U[1][r7] = 1
        U[3][r7] = -1
        V[1*p + k0][r7] = 1
        V[1*p + k1][r7] = 1
        
        # Reconstruction
        # C0k0 = P1 + P4 - P5 + P7
        W[2*k0][r1] = 1
        W[2*k0][r4] = 1
        W[2*k0][r5] = -1
        W[2*k0][r7] = 1
        
        # C1k0 = P2 + P4
        W[2*k0+1][r2] = 1
        W[2*k0+1][r4] = 1
        
        # C0k1 = P3 + P5
        W[2*k1][r3] = 1
        W[2*k1][r5] = 1
        
        # C1k1 = P1 - P2 + P3 + P6
        W[2*k1+1][r1] = 1
        W[2*k1+1][r2] = -1
        W[2*k1+1][r3] = 1
        W[2*k1+1][r6] = 1

    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Trivial block for column 14
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = a * B014
    U[0][r1] = 1
    V[0*p + 14][r1] = 1
    W[2*14][r1] = 1 # C014
    
    # Q2 = b * B114
    U[1][r2] = 1
    V[1*p + 14][r2] = 1
    W[2*14][r2] = 1 # C014
    
    # Q3 = c * B014
    U[2][r3] = 1
    V[0*p + 14][r3] = 1
    W[2*14+1][r3] = 1 # C114
    
    # Q4 = d * B114
    U[3][r4] = 1
    V[1*p + 14][r4] = 1
    W[2*14+1][r4] = 1 # C114
    
    return {"U": U, "V": V, "W": W}
