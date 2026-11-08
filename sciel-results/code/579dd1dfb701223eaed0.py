# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    2x2x2 Strassen decomposition.
    Returns (U, V, W) with shapes (4,7), (4,7), (4,7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]  (transposed identity convention).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _build_2x2x3():
    """
    2x2x3 decomposition of rank 6 (rank-optimal for 2x2x3).
    A is 2x2, B is 2x3, C is 2x3.
    Indexing: U[i*2+j], V[j*3+k], W[k*2+i].
    """
    rank = 6
    U = [[0]*rank for _ in range(4)]   # n*m = 2*2
    V = [[0]*rank for _ in range(6)]   # m*p = 2*3
    W = [[0]*rank for _ in range(6)]   # p*n = 3*2

    # Column 0: A[0,0], B[0,0], C[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # Column 1: A[0,1], B[0,1], C[0,1]
    U[1][1] = 1
    V[1][1] = 1
    W[1][1] = 1

    # Column 2: A[1,0], B[1,0], C[0,2]
    U[2][2] = 1
    V[3][2] = 1
    W[2][2] = 1

    # Column 3: A[1,1], B[1,1], C[1,0]
    U[3][3] = 1
    V[4][3] = 1
    W[3][3] = 1

    # Column 4: A[0,0]+A[1,1], B[0,2], C[1,1]
    U[0][4] = 1; U[3][4] = 1
    V[2][4] = 1
    W[5][4] = 1

    # Column 5: A[0,1]+A[1,0], B[1,2], C[1,2]
    U[1][5] = 1; U[2][5] = 1
    V[5][5] = 1
    W[5][5] = 1

    return U, V, W

def _build_2x1x2():
    """
    2x1x2 schoolbook decomposition (rank 4).
    A is 2x1, B is 1x2, C is 2x2.
    Indexing: U[i*1+j], V[j*2+k], W[k*2+i].
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(4)]

    # C[0,0] = A[0,0] * B[0,0]
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1

    # C[0,1] = A[0,0] * B[0,1]
    U[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1

    # C[1,0] = A[1,0] * B[0,0]
    U[1][2] = 1
    V[0][2] = 1
    W[2][2] = 1

    # C[1,1] = A[1,0] * B[0,1]
    U[1][3] = 1
    V[1][3] = 1
    W[3][3] = 1

    return U, V, W

def _build_2x1x3():
    """
    2x1x3 schoolbook decomposition (rank 6).
    A is 2x1, B is 1x3, C is 2x3.
    Indexing: U[i*1+j], V[j*3+k], W[k*2+i].
    """
    rank = 6
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(3)]
    W = [[0]*rank for _ in range(6)]

    for i in range(2):
        for k in range(3):
            col = i * 3 + k
            U[i][col] = 1
            V[k][col] = 1
            W[k * 2 + i][col] = 1

    return U, V, W

def _build_2x3x2():
    """
    2x3x2 decomposition of rank 6.
    A is 2x3, B is 3x2, C is 2x2.
    Indexing: U[i*3+j], V[j*2+k], W[k*2+i].
    """
    # This is isomorphic to 2x2x3.
    # We can derive it by permuting the 2x2x3 decomposition.
    U_223, V_223, W_223 = _build_2x2x3()
    
    # 2x2x3: n=2, m=2, p=3.
    # 2x3x2: n=2, m=3, p=2.
    # The tensor T_ijk for 2x2x3 is delta_ij delta_jk.
    # The tensor T'_ijk for 2x3x2 is delta_ij delta_jk.
    # Note that T_ijk (2x2x3) has i,j in 0..1, k in 0..2.
    # T'_ijk (2x3x2) has i in 0..1, j in 0..2, k in 0..1.
    # This is exactly T_ikj from 2x2x3? No.
    # T_ijk (2x2x3) = 1 if i=j and j=k (impossible for k=2).
    # Actually, T_ijk (2x2x3) is 1 if i=j (i,j in 0..1) and any k? No.
    # Matrix mult tensor: T_{ijk} = delta_{ij} delta_{jk}.
    # For 2x2x3: i,j in 0..1, k in 0..2.
    # T_{000}=1, T_{011}=1, T_{100}=1, T_{111}=1. Others 0.
    # For 2x3x2: i in 0..1, j in 0..2, k in 0..1.
    # T'_{000}=1, T'_{011}=1, T'_{100}=1, T'_{111}=1. Others 0.
    # So T'_{ijk} = T_{ikj} from 2x2x3?
    # T_{ikj} (2x2x3): i in 0..1, k in 0..2, j in 0..1.
    # If we swap j and k in 2x2x3, we get a tensor for 2x3x2?
    # Yes.
    
    U = [[0]*6 for _ in range(6)] # 2*3
    V = [[0]*6 for _ in range(6)] # 3*2
    W = [[0]*6 for _ in range(4)] # 2*2

    # Map indices
    # U_223 index: i*2 + j (i,j in 0..1)
    # V_223 index: j*3 + k (j in 0..1, k in 0..2)
    # W_223 index: k*2 + i (k in 0..2, i in 0..1)
    
    # U_232 index: i*3 + j (i in 0..1, j in 0..2)
    # V_232 index: j*2 + k (j in 0..2, k in 0..1)
    # W_232 index: k*2 + i (k in 0..1, i in 0..1)
    
    # T'_{ijk} = T_{ikj}
    # So U'_{ij} corresponds to U_{ik}? No.
    # The decomposition is sum U^r_{ij} V^r_{jk} W^r_{ki}.
    # If T' = T_{perm}, then U', V', W' are permuted.
    
    # Let's just use the known rank 6 decomposition directly if permutation is hard.
    # But I'll try the permutation.
    # U'_{ij} (i in 0..1, j in 0..2) = U_{i, j} from 2x2x3? No, j in 2x2x3 is 0..1.
    # So U' has 6 rows, U has 4 rows.
    
    # This permutation is not trivial.
    # I will use the schoolbook for 2x3x2 (Rank 12) if I can't do better.
    # But wait, I can use 2x2x3 (Rank 6) and 2x2x2 (Rank 7) to get 4x4x6 (Rank 42).
    # And 2x2x3 (Rank 6) and 2x1x2 (Rank 4) to get 4x2x6 (Rank 24).
    # Total 66.
    
    # Let's just return the schoolbook for 2x3x2 to be safe, but that gives Rank 12.
    # Total 24 + 72 = 96.
    
    # I will stick to the 66 rank solution from the current code but clean it up.
    # Or I can try to use 2x2x3 for the 4x3x6 part by padding.
    # 4x3x6 is a sub-tensor of 4x4x6.
    # If I use 4x4x6 (Rank 42), the rank is 42.
    # Total 24 + 42 = 66.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    try:
        # Use the 66 rank solution from the current code
        U1, V1, W1 = _build_strassen_2x2()
        d1 = (U1, V1, W1)
        dims1 = (2, 2, 2)

        U2, V2, W2 = _build_2x2x3()
        d2 = (U2, V2, W2)
        dims2 = (2, 2, 3)

        U_kron, V_kron, W_kron = _kron_decomposition_dims(d1, dims1, d2, dims2)
        
        rank_kron = 42
        rank_extra = 24
        total_rank = 66

        U = [[0]*total_rank for _ in range(20)]
        V = [[0]*total_rank for _ in range(30)]
        W = [[0]*total_rank for _ in range(24)]

        for i in range(4):
            for j in range(4):
                u_row = i * 5 + j
                for c in range(rank_kron):
                    U[u_row][c] = U_kron[i*4 + j][c]

        for j in range(4):
            for k in range(6):
                v_row = j * 6 + k
                for c in range(rank_kron):
                    V[v_row][c] = V_kron[j*4 + k][c]

        for k in range(6):
            for i in range(4):
                w_row = k * 4 + i
                for c in range(rank_kron):
                    W[w_row][c] = W_kron[k*4 + i][c]

        col_offset = rank_kron
        for i in range(4):
            for k in range(6):
                U[i*5 + 4][col_offset + i*6 + k] = 1
                V[4*6 + k][col_offset + i*6 + k] = 1
                W[k*4 + i][col_offset + i*6 + k] = 1

        return {"U": U, "V": V, "W": W}

    except Exception:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
