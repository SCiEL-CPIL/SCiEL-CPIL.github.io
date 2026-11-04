"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We use rank 28 (4 blocks of Strassen rank 7)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen block adder
    def add_strassen_block(a_idx, b_idx, c_idx, start_col):
        # a_idx: [a, b, c, d] indices in U
        # b_idx: [e, f, g, h] indices in V
        # c_idx: [x00, x01, x10, x11] indices in W
        
        a, b, c, d = a_idx
        e, f, g, h = b_idx
        x00, x01, x10, x11 = c_idx
        
        # Z1 = (a+d)(e+h)
        col = start_col
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[x00][col] += 1
        W[x11][col] += 1
        
        # Z2 = (c+d)e
        col = start_col + 1
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[x10][col] += 1
        W[x11][col] -= 1
        
        # Z3 = a(f-h)
        col = start_col + 2
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[x01][col] += 1
        W[x11][col] += 1
        
        # Z4 = d(g-e)
        col = start_col + 3
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[x00][col] += 1
        W[x10][col] += 1
        
        # Z5 = (a+b)h
        col = start_col + 4
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[x00][col] -= 1
        W[x01][col] += 1
        
        # Z6 = (c-a)(e+f)
        col = start_col + 5
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[x11][col] += 1
        
        # Z7 = (b-d)(g+h)
        col = start_col + 6
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[x00][col] += 1

    # Block 1: A1 (cols 0,1) * B1,0 (rows 0,1, cols 0,1)
    # A1: a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=4, d=A[1,1]=5
    # B1,0: e=B[0,0]=0, f=B[0,1]=1, g=B[1,0]=4, h=B[1,1]=5
    # C1: x00=C[0,0]=0, x01=C[0,1]=2, x10=C[1,0]=1, x11=C[1,1]=3
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)

    # Block 2: A1 (cols 0,1) * B1,1 (rows 0,1, cols 2,3)
    # A1: same [0, 1, 4, 5]
    # B1,1: e=B[0,2]=2, f=B[0,3]=3, g=B[1,2]=6, h=B[1,3]=7
    # C2: x00=C[0,2]=4, x01=C[0,3]=6, x10=C[1,2]=5, x11=C[1,3]=7
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 7)

    # Block 3: A2 (cols 2,3) * B2,0 (rows 2,3, cols 0,1)
    # A2: a=A[0,2]=2, b=A[0,3]=3, c=A[1,2]=6, d=A[1,3]=7
    # B2,0: e=B[2,0]=8, f=B[2,1]=9, g=B[3,0]=12, h=B[3,1]=13
    # C3: x00=C[0,0]=0, x01=C[0,1]=2, x10=C[1,0]=1, x11=C[1,1]=3
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 14)

    # Block 4: A2 (cols 2,3) * B2,1 (rows 2,3, cols 2,3)
    # A2: same [2, 3, 6, 7]
    # B2,1: e=B[2,2]=10, f=B[2,3]=11, g=B[3,2]=14, h=B[3,3]=15
    # C4: x00=C[0,2]=4, x01=C[0,3]=6, x10=C[1,2]=5, x11=C[1,3]=7
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
