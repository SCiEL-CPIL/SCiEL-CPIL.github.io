"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing ranks of blocks.
    # Block 1: A1 (2x2) * B1 (2x2) -> C11 (2x2). Rank 7.
    # Block 2: A1 (2x2) * B2 (2x3) -> C12 (2x3). Rank 11.
    # Block 3: A2 (2x3) * B3 (3x2) -> C21 (2x2). Rank 11.
    # Block 4: A2 (2x3) * B4 (3x3) -> C22 (2x3). Rank 13.
    # Total Rank = 7 + 11 + 11 + 13 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_block(u_local, v_local, w_local, i_off, j_off, k_off, n_loc, m_loc, p_loc):
        nonlocal current_rank
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    val_u = u_local[i][j]
                    val_v = v_local[j][k]
                    val_w = w_local[k][i]
                    if val_u != 0 and val_v != 0 and val_w != 0:
                        U[i_off + i][current_rank] += val_u
                        V[j_off + j][current_rank] += val_v
                        W[k_off + k][current_rank] += val_w
        current_rank += 1
        # Note: The above logic is slightly wrong for tensor decomposition.
        # A term is U_i,j * V_j,k * W_k,i.
        # We need to add a rank-1 term for each non-zero combination?
        # No, a rank-1 term in the decomposition is defined by vectors u, v, w.
        # The code above adds to the same column for all i,j,k?
        # This is incorrect.
        
        # Correct approach:
        # A rank-r decomposition is a sum of r rank-1 tensors.
        # Each rank-1 tensor is defined by u (n*m), v (m*p), w (p*n).
        # We need to generate these vectors.
        
        pass

    # Let's rewrite the construction properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a set of rank-1 terms
    def add_terms(terms, i_off, j_off, k_off):
        nonlocal col
        for u_vec, v_vec, w_vec in terms:
            for idx, val in enumerate(u_vec):
                if val != 0:
                    U[i_off + idx // len(u_vec[0]) if False else idx][col] += val # Need proper indexing
            # This is getting messy with flat indices.
            # Let's use 2D lists for local blocks and then map.
            pass

    # Simpler: Generate the full U, V, W for each block type, then place them.
    
    def get_2x2x2_strassen():
        # Returns list of (U, V, W) where U is 2x2, V is 2x2, W is 2x2
        # U[i,j] is coeff for A[i,j]
        # V[j,k] is coeff for B[j,k]
        # W[k,i] is coeff for C[k,i] (Note: W index is k*n+i)
        terms = []
        # M1 = A11(B11+B22)
        terms.append(([ [1,0],[0,0] ], [ [1,0],[0,1] ], [ [1,0],[0,1] ]))
        # M2 = (A11+A21)B22
        terms.append(([ [1,0],[1,0] ], [ [0,0],[0,1] ], [ [0,1],[1,0] ])) # C21 gets M2? 
        # Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5 - M6 - M7
        # C21 = M2 + M6 - M1 - M3
        # C22 = M1 + M3 - M2 + M5
        # W is p*n = 2*2. W[k,i].
        # C11 is i=0, k=0. W[0,0].
        # C12 is i=0, k=1. W[1,0].
        # C21 is i=1, k=0. W[0,1].
        # C22 is i=1, k=1. W[1,1].
        
        # M1: C11 (+1), C22 (+1).
        terms[0] = ([ [1,0],[0,0] ], [ [1,0],[0,1] ], [ [1,0],[0,1] ])
        # M2: C21 (+1).
        terms[1] = ([ [1,0],[1,0] ], [ [0,0],[0,1] ], [ [0,1],[0,0] ])
        # M3: C12 (+1), C22 (-1).
        terms[2] = ([ [0,0],[0,1] ], [ [1,1],[0,0] ], [ [0,1],[-1,0] ]) # A22(B11+B12)
        # M4: C11 (+1).
        terms[3] = ([ [0,0],[1,0] ], [ [1,0],[0,0] ], [ [1,0],[0,0] ]) # (A22+A12)B11 -> A22 B11 + A12 B11?
        # M4 = (A22+A12)B11.
        # M5: C11 (-1), C12 (+1), C22 (+1).
        terms[4] = ([ [0,1],[1,0] ], [ [0,0],[1,1] ], [ [-1,1],[0,1] ]) # (A12+A22)(B21+B22)
        # M6: C12 (-1), C21 (+1).
        terms[5] = ([ [1,1],[0,0] ], [ [0,-1],[0,1] ], [ [0,-1],[1,0] ]) # (A11+A12)(B22-B12)
        # M7: C11 (+1), C12 (-1).
        terms[6] = ([ [0,0],[1,1] ], [ [1,0],[-1,0] ], [ [1,-1],[0,0] ]) # (A21+A22)(B11-B21)
        
        return terms

    def get_2x2x1():
        # A (2x2), B (2x1), C (2x1)
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        terms = []
        # Term 1: A11 B1 -> C1
        terms.append(([ [1,0],[0,0] ], [ [1],[0] ], [ [1],[0] ]))
        # Term 2: A12 B2 -> C1
        terms.append(([ [0,1],[0,0] ], [ [0],[1] ], [ [1],[0] ]))
        # Term 3: A21 B1 -> C2
        terms.append(([ [0,0],[1,0] ], [ [1],[0] ], [ [0],[1] ]))
        # Term 4: A22 B2 -> C2
        terms.append(([ [0,0],[0,1] ], [ [0],[1] ], [ [0],[1] ]))
        return terms

    def get_2x3x3():
        # Use additive: 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (1) + 2x1x1 (1) = 13
        # Split A into A1 (2x2), A2 (2x1)
        # Split B into B1 (2x2), B2 (2x1), B3 (1x2), B4 (1x1)
        # C = A1 B1 + A1 B2 + A2 B3 + A2 B4
        
        terms = []
        
        # Block A1 B1 (2x2x2)
        # Map indices: A1 is A[0:2, 0:2], B1 is B[0:2, 0:2], C11 is C[0:2, 0:2]
        # We need to embed 2x2 terms into 2x3 and 3x3.
        for u, v, w in get_2x2x2_strassen():
            u_full = [ [u[0][0], u[0][1], 0], [u[1][0], u[1][1], 0] ]
            v_full = [ [v[0][0], v[0][1], 0], [v[1][0], v[1][1], 0], [0,0,0] ]
            w_full = [ [w[0][0], w[0][1], 0], [w[1][0], w[1][1], 0], [0,0,0] ] # W is 3x2 (k,i)
            terms.append((u_full, v_full, w_full))
            
        # Block A1 B2 (2x2x1)
        # A1 (2x2), B2 (2x1) [B[0:2, 2]], C12 (2x1) [C[0:2, 2]]
        for u, v, w in get_2x2x1():
            u_full = [ [u[0][0], u[0][1], 0], [u[1][0], u[1][1], 0] ]
            v_full = [ [0,0,v[0][0]], [0,0,v[1][0]], [0,0,0] ]
            w_full = [ [0,0], [0,0], [w[0][0], w[1][0]] ] # W[k,i]. k=2 is 3rd row.
            # W is 3x2. Row 0: k=0. Row 1: k=1. Row 2: k=2.
            # C12 is i=0,1 and k=2.
            # So w_full[2][0] = w[0][0], w_full[2][1] = w[1][0].
            w_full = [ [0,0], [0,0], [w[0][0], w[1][0]] ]
            terms.append((u_full, v_full, w_full))
            
        # Block A2 B3 (2x1x2)
        # A2 (2x1) [A[0:2, 2]], B3 (1x2) [B[2, 0:2]], C21 (2x2) [C[0:2, 0:2]]
        # C[i,k] = A[i,2] B[2,k]
        # Rank 1 term: U[0,2]=1, U[1,2]=1? No.
        # U[i,j] is coeff for A[i,j].
        # We need U[0,2]=1, U[1,2]=1?
        # Term: U = [[0,0,1],[0,0,1]], V = [[1,1],[0,0],[0,0]], W = [[1,1],[1,1],[0,0]]?
        # C00 = A02 B20. C01 = A02 B21.
        # C10 = A12 B20. C11 = A12 B21.
        # This is 1 rank term?
        # T_{i,j,k} = 1 if j=2.
        # U[0,2]=1, U[1,2]=1.
        # V[2,0]=1, V[2,1]=1.
        # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1.
        # Yes, rank 1.
        u_full = [ [0,0,1], [0,0,1] ]
        v_full = [ [0,0,0], [0,0,0], [1,1,0] ]
        w_full = [ [1,1], [1,1], [0,0] ]
        terms.append((u_full, v_full, w_full))
        
        # Block A2 B4 (2x1x1)
        # A2 (2x1), B4 (1x1) [B[2,2]], C22 (2x1) [C[0:2, 2]]
        # C[i,2] = A[i,2] B[2,2]
        u_full = [ [0,0,1], [0,0,1] ]
        v_full = [ [0,0,0], [0,0,0], [0,0,1] ]
        w_full = [ [0,0], [0,0], [1,1] ]
        terms.append((u_full, v_full, w_full))
        
        return terms

    # Assemble
    # Block 1: 2x2x2 at (0,0,0)
    for u, v, w in get_2x2x2_strassen():
        for i in range(2):
            for j in range(2):
                if u[i][j] != 0:
                    U[i*5+j][col] += u[i][j]
        for j in range(2):
            for k in range(2):
                if v[j][k] != 0:
                    V[j*6+k][col] += v[j][k]
        for k in range(2):
            for i in range(2):
                if w[k][i] != 0:
                    W[k*2+i][col] += w[k][i]
        col += 1

    # Block 2: 2x2x3 at (0,0,2)
    # A1 (2x2), B2 (2x3) [B[0:2, 2:5]], C12 (2x3) [C[0:2, 2:5]]
    # Use 2x2x2 (7) + 2x2x1 (4)
    # 2x2x2 part: B[0:2, 2:4], C[0:2, 2:4]
    for u, v, w in get_2x2x2_strassen():
        for i in range(2):
            for j in range(2):
                if u[i][j] != 0:
                    U[i*5+j][col] += u[i][j]
        for j in range(2):
            for k in range(2):
                if v[j][k] != 0:
                    V[j*6 + (2+k)][col] += v[j][k]
        for k in range(2):
            for i in range(2):
                if w[k][i] != 0:
                    W[(2+k)*2+i][col] += w[k][i]
        col += 1

    # 2x2x1 part: B[0:2, 4], C[0:2, 4]
    for u, v, w in get_2x2x1():
        for i in range(2):
            for j in range(2):
                if u[i][j] != 0:
                    U[i*5+j][col] += u[i][j]
        for j in range(2):
            if v[j][0] != 0:
                V[j*6 + 4][col] += v[j][0]
        for k in range(1):
            for i in range(2):
                if w[k][i] != 0:
                    W[(4+k)*2+i][col] += w[k][i]
        col += 1

    # Block 3: 2x3x2 at (2,0,0)
    # A2 (2x3) [A[0:2, 2:5]], B3 (3x2) [B[2:5, 0:2]], C21 (2x2) [C[0:2, 0:2]]
    # Use 2x2x2 (7) + 2x1x2 (4)? No, 2x3x2 = 2x2x2 + 2x1x2.
    # 2x2x2: A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2]
    for u, v, w in get_2x2x2_strassen():
        for i in range(2):
            for j in range(2):
                if u[i][j] != 0:
                    U[i*5 + (2+j)][col] += u[i][j]
        for j in range(2):
            for k in range(2):
                if v[j][k] != 0:
                    V[(2+j)*6 + k][col] += v[j][k]
        for k in range(2):
            for i in range(2):
                if w[k][i] != 0:
                    W[k*2+i][col] += w[k][i]
        col += 1

    # 2x1x2: A[0:2, 4], B[4, 0:2], C[0:2, 0:2]
    # A (2x1), B (1x2), C (2x2). Rank 4?
    # C[i,k] = A[i,4] B[4,k].
    # This is rank 1?
    # T_{i,j,k} with j=4.
    # U[0,4]=1, U[1,4]=1.
    # V[4,0]=1, V[4,1]=1.
    # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1.
    # Yes, rank 1.
    u_full = [ [0,0,0,0,1], [0,0,0,0,1] ]
    v_full = [ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [1,1,0,0,0,0], [0,0,0,0,0,0] ]
    w_full = [ [1,1], [1,1], [0,0], [0,0], [0,0], [0,0] ]
    # Add this term
    for i in range(2):
        if u_full[i][4] != 0:
            U[i*5+4][col] += u_full[i][4]
    for k in range(2):
        if v_full[4][k] != 0:
            V[4*6+k][col] += v_full[4][k]
    for k in range(2):
        for i in range(2):
            if w_full[k][i] != 0:
                W[k*2+i][col] += w_full[k][i]
    col += 1

    # Block 4: 2x3x3 at (2,2,2)
    # A2 (2x3) [A[0:2, 2:5]], B4 (3x3) [B[2:5, 2:5]], C22 (2x3) [C[0:2, 2:5]]
    # Use 13-rank decomposition from get_2x3x3
    for u, v, w in get_2x3x3():
        for i in range(2):
            for j in range(3):
                if u[i][j] != 0:
                    U[i*5 + (2+j)][col] += u[i][j]
        for j in range(3):
            for k in range(3):
                if v[j][k] != 0:
                    V[(2+j)*6 + (2+k)][col] += v[j][k]
        for k in range(3):
            for i in range(2):
                if w[k][i] != 0:
                    W[(2+k)*2+i][col] += w[k][i]
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
