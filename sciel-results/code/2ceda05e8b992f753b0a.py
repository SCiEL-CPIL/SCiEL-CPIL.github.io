"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(U_off, V_off, W_off):
        nonlocal col
        # U_off: (i, j) -> U[i*m + j]
        # V_off: (j, k) -> V[j*p + k]
        # W_off: (k, i) -> W[k*n + i]
        
        # M1 = (a11+a22)(b11+b22)
        U[U_off(0,0)][col] = 1
        U[U_off(1,1)][col] = 1
        V[V_off(0,0)][col] = 1
        V[V_off(1,1)][col] = 1
        W[W_off(0,0)][col] = 1
        W[W_off(1,1)][col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[U_off(1,0)][col] = 1
        U[U_off(1,1)][col] = 1
        V[V_off(0,0)][col] = 1
        W[W_off(1,0)][col] = 1 # c21 (i=1, k=1? No. c21 is i=1, k=0. W index k*n+i = 0*2+1=1)
        # Wait, W_off(k,i).
        # c12 is i=0, k=1. W index 1*2+0=2.
        # c21 is i=1, k=0. W index 0*2+1=1.
        # c22 is i=1, k=1. W index 1*2+1=3.
        # c11 is i=0, k=0. W index 0*2+0=0.
        
        # Let's re-verify W indices for Strassen
        # c11 (i=0,k=0) -> W[0]
        # c21 (i=1,k=0) -> W[1]
        # c12 (i=0,k=1) -> W[2]
        # c22 (i=1,k=1) -> W[3]
        
        # M2: c12=1, c21=1
        W[W_off(1,0)][col] = 1 # c12
        W[W_off(0,1)][col] = 1 # c21
        col += 1
        
        # M3 = a11(b21-b22)
        U[U_off(0,0)][col] = 1
        V[V_off(1,0)][col] = 1
        V[V_off(1,1)][col] = -1
        W[W_off(1,0)][col] = 1 # c12
        W[W_off(0,1)][col] = -1 # c21
        col += 1
        
        # M4 = a22(b12-b11)
        U[U_off(1,1)][col] = 1
        V[V_off(0,1)][col] = 1
        V[V_off(0,0)][col] = -1
        W[W_off(0,0)][col] = 1 # c11
        W[W_off(1,1)][col] = 1 # c22
        col += 1
        
        # M5 = (a11+a12)b22
        U[U_off(0,0)][col] = 1
        U[U_off(0,1)][col] = 1
        V[V_off(1,1)][col] = 1
        W[W_off(0,0)][col] = -1 # c11
        W[W_off(0,1)][col] = 1 # c21
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[U_off(1,0)][col] = 1
        U[U_off(0,0)][col] = -1
        V[V_off(0,0)][col] = 1
        V[V_off(0,1)][col] = 1
        W[W_off(1,0)][col] = 1 # c12
        W[W_off(0,1)][col] = -1 # c21
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[U_off(0,1)][col] = 1
        U[U_off(1,1)][col] = -1
        V[V_off(1,0)][col] = 1
        V[V_off(1,1)][col] = 1
        W[W_off(0,0)][col] = 1 # c11
        W[W_off(1,0)][col] = -1 # c12
        col += 1

    def add_schoolbook_221(U_off, V_off, W_off):
        nonlocal col
        # A (2x2), B (2x1), C (2x1)
        # c1 = a11 b1 + a12 b2
        # c2 = a21 b1 + a22 b2
        
        # T1: a11 b1 -> c1
        U[U_off(0,0)][col] = 1
        V[V_off(0,0)][col] = 1
        W[W_off(0,0)][col] = 1
        col += 1
        
        # T2: a12 b2 -> c1
        U[U_off(0,1)][col] = 1
        V[V_off(1,0)][col] = 1
        W[W_off(0,0)][col] = 1
        col += 1
        
        # T3: a21 b1 -> c2
        U[U_off(1,0)][col] = 1
        V[V_off(0,0)][col] = 1
        W[W_off(0,1)][col] = 1
        col += 1
        
        # T4: a22 b2 -> c2
        U[U_off(1,1)][col] = 1
        V[V_off(1,0)][col] = 1
        W[W_off(0,1)][col] = 1
        col += 1

    def add_rank7_217(U_off, V_off, W_off):
        nonlocal col
        # A (2x1), B (1x7), C (2x7)
        # c_i,k = a_i b_k
        # Use 7 terms.
        for k in range(7):
            # U is 2x1. Indices (i, 0)
            U[U_off(0,0)][col] = 1
            U[U_off(1,0)][col] = 1
            # V is 1x7. Indices (0, k)
            V[V_off(0,k)][col] = 1
            # W is 7x2. Indices (k, i)
            W[W_off(k,0)][col] = 1
            W[W_off(k,1)][col] = 1
            col += 1

    # Block 1: A_top (2x2) * B_top_L (2x3)
    # Split B_top_L into B_LL (2x2) and B_LR (2x1)
    
    # 1a: A_top * B_LL (Strassen)
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..1
    # C indices: k in 0..1, i in 0..1
    U_off_1a = lambda i,j: i*m + j
    V_off_1a = lambda j,k: j*p + k
    W_off_1a = lambda k,i: k*n + i
    add_strassen(U_off_1a, V_off_1a, W_off_1a)
    
    # 1b: A_top * B_LR (Schoolbook 221)
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 2..2 (local 0)
    # C indices: k in 2..2 (local 0), i in 0..1
    U_off_1b = lambda i,j: i*m + j
    V_off_1b = lambda j,k: j*p + (2+k)
    W_off_1b = lambda k,i: (2+k)*n + i
    add_schoolbook_221(U_off_1b, V_off_1b, W_off_1b)

    # Block 2: A_top (2x2) * B_top_R (2x4)
    # Split B_top_R into B_RL (2x2) and B_RR (2x2)
    
    # 2a: A_top * B_RL (Strassen)
    # B indices: j in 0..1, k in 3..4 (local 0..1)
    # C indices: k in 3..4 (local 0..1), i in 0..1
    U_off_2a = lambda i,j: i*m + j
    V_off_2a = lambda j,k: j*p + (3+k)
    W_off_2a = lambda k,i: (3+k)*n + i
    add_strassen(U_off_2a, V_off_2a, W_off_2a)
    
    # 2b: A_top * B_RR (Strassen)
    # B indices: j in 0..1, k in 5..6 (local 0..1)
    # C indices: k in 5..6 (local 0..1), i in 0..1
    U_off_2b = lambda i,j: i*m + j
    V_off_2b = lambda j,k: j*p + (5+k)
    W_off_2b = lambda k,i: (5+k)*n + i
    add_strassen(U_off_2b, V_off_2b, W_off_2b)

    # Block 3: A_bot (2x1) * B_bot (1x7)
    # A indices: i in 0..1, j in 2..2
    # B indices: j in 2..2, k in 0..6
    # C indices: k in 0..6, i in 0..1
    U_off_3 = lambda i,j: i*m + 2
    V_off_3 = lambda j,k: 2*p + k
    W_off_3 = lambda k,i: k*n + i
    add_rank7_217(U_off_3, V_off_3, W_off_3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
