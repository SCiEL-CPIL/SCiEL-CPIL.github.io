import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for 2x2x2 multiplication
    # a,b,c,d are indices in U for A (2x2 block)
    # e,f,g,h are indices in V for B (2x2 block)
    # w_idx is the base index for W (2x2 block output)
    # We need to add coefficients to U, V, W for the 7 products
    
    # Strassen products:
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
    
    def add_strassen_block(u_idx, v_idx, w_idx, start_col):
        nonlocal col
        if time.time() > deadline:
            return
        
        # u_idx: [a, b, c, d] indices in U
        # v_idx: [e, f, g, h] indices in V
        # w_idx: [c00, c01, c10, c11] indices in W
        
        a, b, c, d = u_idx
        e, f, g, h = v_idx
        c00, c01, c10, c11 = w_idx
        
        # M1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1

    # Indices for A (2x4)
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # Indices for B (4x5)
    # B[j,k] = j*5 + k
    
    # Indices for W (5x2) -> W[k*n + i] = k*2 + i
    # C[i,k] corresponds to W[k*2 + i]
    
    # Block 1: Cols 0,1
    # A1: A[0,0], A[0,1], A[1,0], A[1,1] -> [0, 1, 4, 5]
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> [0, 1, 5, 6]
    # W for C[:, 0:2]:
    # C[0,0]->W[0], C[0,1]->W[1], C[1,0]->W[2], C[1,1]->W[3]
    add_strassen_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 2, 3], 0)
    
    # A2: A[0,2], A[0,3], A[1,2], A[1,3] -> [2, 3, 6, 7]
    # B21: B[2,0], B[2,1], B[3,0], B[3,1] -> [10, 11, 15, 16]
    # W for C[:, 0:2]: Same W indices [0, 1, 2, 3]
    add_strassen_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 1, 2, 3], 0)
    
    # Block 2: Cols 2,3
    # A1: [0, 1, 4, 5]
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> [2, 3, 7, 8]
    # W for C[:, 2:4]:
    # C[0,2]->W[4], C[0,3]->W[5], C[1,2]->W[6], C[1,3]->W[7]
    add_strassen_block([0, 1, 4, 5], [2, 3, 7, 8], [4, 5, 6, 7], 0)
    
    # A2: [2, 3, 6, 7]
    # B22: B[2,2], B[2,3], B[3,2], B[3,3] -> [12, 13, 17, 18]
    # W for C[:, 2:4]: Same W indices [4, 5, 6, 7]
    add_strassen_block([2, 3, 6, 7], [12, 13, 17, 18], [4, 5, 6, 7], 0)
    
    # Part 2: Col 4 (Last column)
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4] + A[0,2]B[2,4] + A[0,3]B[3,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4] + A[1,2]B[2,4] + A[1,3]B[3,4]
    
    # W indices for C[:, 4]:
    # C[0,4] -> W[4*2 + 0] = W[8]
    # C[1,4] -> W[4*2 + 1] = W[9]
    
    # Terms for C[0,4]
    # A[0,0]=0, B[0,4]=4
    U[0][col] += 1; V[4][col] += 1; W[8][col] += 1; col += 1
    # A[0,1]=1, B[1,4]=9
    U[1][col] += 1; V[9][col] += 1; W[8][col] += 1; col += 1
    # A[0,2]=2, B[2,4]=14
    U[2][col] += 1; V[14][col] += 1; W[8][col] += 1; col += 1
    # A[0,3]=3, B[3,4]=19
    U[3][col] += 1; V[19][col] += 1; W[8][col] += 1; col += 1
    
    # Terms for C[1,4]
    # A[1,0]=4, B[0,4]=4
    U[4][col] += 1; V[4][col] += 1; W[9][col] += 1; col += 1
    # A[1,1]=5, B[1,4]=9
    U[5][col] += 1; V[9][col] += 1; W[9][col] += 1; col += 1
    # A[1,2]=6, B[2,4]=14
    U[6][col] += 1; V[14][col] += 1; W[9][col] += 1; col += 1
    # A[1,3]=7, B[3,4]=19
    U[7][col] += 1; V[19][col] += 1; W[9][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
