"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition: 
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # Split B (3x6) into B_left (2x6) and B_right (1x6).
    # C = A_left * B_left + A_right * B_right
    #
    # Part 1: A_left (2x2) * B_left (2x6). 
    #   B_left consists of 3 blocks of 2x2 matrices: B0, B1, B2.
    #   Use Strassen for each 2x2x2x2 block. Rank = 3 * 7 = 21.
    #
    # Part 2: A_right (2x1) * B_right (1x6).
    #   This is 2 columns of A (scalar per row) times 6 rows of B.
    #   Rank = 2 * 1 * 6 = 12.
    #
    # Total Rank = 33.
    
    n, m, p = DIMENSIONS
    U = []
    V = []
    W = []
    
    # Helper to append a term (u, v, w) to the lists
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Strassen 2x2 * 2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices for A: 0,1,2,3 corresponding to a,b,c,d
    # V indices for B: 0,1,2,3 corresponding to e,f,g,h
    # W indices for C: 0,1,2,3 corresponding to C11,C12,C21,C22
    
    def strassen_terms(a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, c11_idx, c12_idx, c21_idx, c22_idx):
        # M1
        add_term([a_idx, d_idx], [e_idx, h_idx], [c11_idx, c22_idx]) # W: +1, +1
        # Actually W[k*n+i] so we need to map C indices to W indices later.
        # Let's define W basis vectors explicitly to avoid confusion.
        pass # We will implement manually below.

    # Let's build the matrices row by row.
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3 Strassen blocks
    # A_left indices in U: 
    # Row 0 (i=0): A[0,0]=idx0, A[0,1]=idx1
    # Row 1 (i=1): A[1,0]=idx2, A[1,1]=idx3
    # Note: U shape is (n*m, rank). Index is i*m + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 2. i=1, j=1 -> 3.
    
    # B_left indices in V:
    # B_left is 2x6. Rows j=0,1. Cols k=0..5.
    # V index is j*p + k.
    # Block 0: k=0,1. V indices: 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7. (e,f,g,h for block 0)
    # Block 1: k=2,3. V indices: 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9.
    # Block 2: k=4,5. V indices: 0*6+4=4, 0*6+5=5, 1*6+4=10, 1*6+5=11.
    
    # W indices for C (2x6). Index is k*n + i.
    # For Block 0 (cols 0,1):
    # C[0,0] -> k=0, i=0 -> 0*2+0=0
    # C[0,1] -> k=1, i=0 -> 1*2+0=2
    # C[1,0] -> k=0, i=1 -> 0*2+1=1
    # C[1,1] -> k=1, i=1 -> 1*2+1=3
    
    # For Block 1 (cols 2,3):
    # C[0,2] -> 2*2+0=4
    # C[0,3] -> 3*2+0=6
    # C[1,2] -> 2*2+1=5
    # C[1,3] -> 3*2+1=7
    
    # For Block 2 (cols 4,5):
    # C[0,4] -> 4*2+0=8
    # C[0,5] -> 5*2+0=10
    # C[1,4] -> 4*2+1=9
    # C[1,5] -> 5*2+1=11
    
    def add_strassen_block(v_e, v_f, v_g, v_h, w_c11, w_c12, w_c21, w_c22):
        nonlocal col
        # M1 = (a+d)(e+h)
        # a=U[0], d=U[3], e=V[v_e], h=V[v_h]
        # W: C11 += M1, C22 += M1
        U[0][col] = 1.0; U[3][col] = 1.0
        V[v_e][col] = 1.0; V[v_h][col] = 1.0
        W[w_c11][col] = 1.0; W[w_c22][col] = 1.0
        col += 1
        
        # M2 = (c+d)e
        # c=U[2], d=U[3], e=V[v_e]
        # W: C21 += M2
        U[2][col] = 1.0; U[3][col] = 1.0
        V[v_e][col] = 1.0
        W[w_c21][col] = 1.0
        col += 1
        
        # M3 = a(f-h)
        # a=U[0], f=V[v_f], h=V[v_h]
        # W: C12 += M3, C22 += M3
        U[0][col] = 1.0
        V[v_f][col] = 1.0; V[v_h][col] = -1.0
        W[w_c12][col] = 1.0; W[w_c22][col] = 1.0
        col += 1
        
        # M4 = d(g-e)
        # d=U[3], g=V[v_g], e=V[v_e]
        # W: C11 += M4, C21 += M4
        U[3][col] = 1.0
        V[v_g][col] = 1.0; V[v_e][col] = -1.0
        W[w_c11][col] = 1.0; W[w_c21][col] = 1.0
        col += 1
        
        # M5 = (a+b)h
        # a=U[0], b=U[1], h=V[v_h]
        # W: C11 -= M5, C12 += M5
        U[0][col] = 1.0; U[1][col] = 1.0
        V[v_h][col] = 1.0
        W[w_c11][col] = -1.0; W[w_c12][col] = 1.0
        col += 1
        
        # M6 = (c-a)(e+f)
        # c=U[2], a=U[0], e=V[v_e], f=V[v_f]
        # W: C22 += M6
        U[2][col] = 1.0; U[0][col] = -1.0
        V[v_e][col] = 1.0; V[v_f][col] = 1.0
        W[w_c22][col] = 1.0
        col += 1
        
        # M7 = (b-d)(g+h)
        # b=U[1], d=U[3], g=V[v_g], h=V[v_h]
        # W: C11 += M7
        U[1][col] = 1.0; U[3][col] = -1.0
        V[v_g][col] = 1.0; V[v_h][col] = 1.0
        W[w_c11][col] = 1.0
        col += 1

    # Block 0: V[0,1,6,7], W[0,2,1,3]
    add_strassen_block(0, 1, 6, 7, 0, 2, 1, 3)
    # Block 1: V[2,3,8,9], W[4,6,5,7]
    add_strassen_block(2, 3, 8, 9, 4, 6, 5, 7)
    # Block 2: V[4,5,10,11], W[8,10,9,11]
    add_strassen_block(4, 5, 10, 11, 8, 10, 9, 11)
    
    # Part 2: A_right (2x1) * B_right (1x6)
    # A_right is column 2 of A. Indices in U: i=0,j=2 -> 0*3+2=4; i=1,j=2 -> 1*3+2=5.
    # B_right is row 2 of B. Indices in V: j=2, k=0..5 -> 2*6+0=12 ... 2*6+5=17.
    # C contributions:
    # C[0, k] += A[0,2] * B[2,k] -> W index k*2+0
    # C[1, k] += A[1,2] * B[2,k] -> W index k*2+1
    
    for k in range(p):
        v_idx = 2 * p + k
        w_idx_0 = k * n + 0
        w_idx_1 = k * n + 1
        
        # Term for A[0,2] * B[2,k]
        U[4][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx_0][col] = 1.0
        col += 1
        
        # Term for A[1,2] * B[2,k]
        U[5][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx_1][col] = 1.0
        col += 1
        
    # Verify rank used
    assert col == rank, f"Expected rank {rank}, used {col}"
    
    # Convert to list of lists of integers if possible, otherwise floats
    # The problem asks for integer/half-integer entries.
    # Our entries are -1, 0, 1.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
