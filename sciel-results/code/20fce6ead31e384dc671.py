"""Matrix multiplication 4x5x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U rows: (i,j) -> i*2+j for A
    V rows: (j,k) -> j*2+k for B
    W rows: (k,i) -> k*2+i for C
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

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m * m x p -> n x p.
    Rank = n*m*p.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _decomp_block(n_sub, m_sub, p_sub, rank_sub, U_sub, V_sub, W_sub,
                  i_s, j_s, k_s, n, m, p, U, V, W, offset):
    """
    Place a sub-decomposition for block A[i_s:i_s+n_sub, j_s:j_s+m_sub] *
    B[j_s:j_s+m_sub, k_s:k_s+p_sub] -> C[i_s:i_s+n_sub, k_s:k_s+p_sub]
    into the global matrices at column offset.
    """
    for r in range(n_sub * m_sub):
        i_loc = r // m_sub
        j_loc = r % m_sub
        i_glob = i_s + i_loc
        j_glob = j_s + j_loc
        r_glob = i_glob * m + j_glob
        for c in range(rank_sub):
            U[r_glob][offset + c] = U_sub[r][c]
    
    for r in range(m_sub * p_sub):
        j_loc = r // p_sub
        k_loc = r % p_sub
        j_glob = j_s + j_loc
        k_glob = k_s + k_loc
        r_glob = j_glob * p + k_glob
        for c in range(rank_sub):
            V[r_glob][offset + c] = V_sub[r][c]
    
    for r in range(p_sub * n_sub):
        k_loc = r // n_sub
        i_loc = r % n_sub
        k_glob = k_s + k_loc
        i_glob = i_s + i_loc
        r_glob = k_glob * n + i_glob
        for c in range(rank_sub):
            W[r_glob][offset + c] = W_sub[r][c]

def _build_strassen_4x4():
    """
    Build 4x4x4 Strassen decomposition (rank 49).
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    rank44 = 49
    U44 = [[0]*rank44 for _ in range(16)]
    V44 = [[0]*rank44 for _ in range(16)]
    W44 = [[0]*rank44 for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U44[u_row][col] += coeff_a * sigma
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V44[v_row][col] += coeff_b * sigma
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W44[w_row][col] += coeff_c * sigma
    
    return U44, V44, W44

def _build_strassen_4x4x2():
    """
    Build 4x4x2 Strassen decomposition (rank 28).
    A is 4x4, B is 4x2, C is 4x2.
    U shape: (16, 28), V shape: (8, 28), W shape: (8, 28)
    """
    U2, V2, W2 = _strassen_2x2()
    rank442 = 28
    U442 = [[0]*rank442 for _ in range(16)]
    V442 = [[0]*rank442 for _ in range(8)]
    W442 = [[0]*rank442 for _ in range(8)]
    
    block_positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for idx, (bi, bj) in enumerate(block_positions):
        base = idx * 7
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = bi * 2 + li
                        global_j = bj * 2 + lj
                        u_row = global_i * 4 + global_j
                        U442[u_row][col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = bj * 2 + lj
                        global_k = lk
                        v_row = global_j * 2 + global_k
                        V442[v_row][col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = lk
                        global_i = bi * 2 + li
                        w_row = global_k * 4 + global_i
                        W442[w_row][col] += coeff_c
    
    return U442, V442, W442

def _build_strassen_2x3():
    """
    Build 2x3x2 Strassen-like decomposition.
    A is 2x3, B is 3x2, C is 2x2.
    Split m=3 into 2+1.
    U shape: (6, rank), V shape: (6, rank), W shape: (4, rank)
    """
    U2, V2, W2 = _strassen_2x2()
    U21, V21, W21 = _schoolbook(2, 1, 2)
    
    rank = 7 + 4  # 2x2x2 Strassen + 2x1x2 schoolbook
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(4)]
    
    # Place 2x2x2 Strassen for A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(7):
            U[i_loc*3 + j_loc][c] = U2[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(7):
            V[j_loc*2 + k_loc][c] = V2[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(7):
            W[k_loc*2 + i_loc][c] = W2[r][c]
    
    # Place 2x1x2 schoolbook for A[0:2,2:3]*B[2:3,0:2]->C[0:2,0:2]
    for r in range(6):
        i_loc = r // 2
        j_loc = r % 2  # j_loc is always 0 here since m_sub=1
        for c in range(4):
            U[i_loc*3 + 2][7+c] = U21[r][c]
    for r in range(4):
        j_loc = r // 2  # j_loc is 0
        k_loc = r % 2
        for c in range(4):
            V[2*2 + k_loc][7+c] = V21[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(4):
            W[k_loc*2 + i_loc][7+c] = W21[r][c]
    
    return U, V, W

def _build_strassen_3x2():
    """
    Build 3x2x2 Strassen-like decomposition.
    A is 3x2, B is 2x2, C is 3x2.
    Split n=3 into 2+1.
    U shape: (6, rank), V shape: (4, rank), W shape: (6, rank)
    """
    U2, V2, W2 = _strassen_2x2()
    U12, V12, W12 = _schoolbook(1, 2, 2)
    
    rank = 7 + 4  # 2x2x2 Strassen + 1x2x2 schoolbook
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(6)]
    
    # Place 2x2x2 Strassen for A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(7):
            U[i_loc*2 + j_loc][c] = U2[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(7):
            V[j_loc*2 + k_loc][c] = V2[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(7):
            W[k_loc*3 + i_loc][c] = W2[r][c]
    
    # Place 1x2x2 schoolbook for A[2:3,0:2]*B[0:2,0:2]->C[2:3,0:2]
    for r in range(4):
        i_loc = 0
        j_loc = r // 2
        for c in range(4):
            U[2*2 + j_loc][7+c] = U12[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(4):
            V[j_loc*2 + k_loc][7+c] = V12[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = 2
        for c in range(4):
            W[k_loc*3 + i_loc][7+c] = W12[r][c]
    
    return U, V, W

def _build_strassen_2x2x3():
    """
    Build 2x2x3 Strassen-like decomposition.
    A is 2x2, B is 2x3, C is 2x3.
    Split p=3 into 2+1.
    U shape: (4, rank), V shape: (6, rank), W shape: (6, rank)
    """
    U2, V2, W2 = _strassen_2x2()
    U221, V221, W221 = _schoolbook(2, 2, 1)
    
    rank = 7 + 4  # 2x2x2 Strassen + 2x2x1 schoolbook
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Place 2x2x2 Strassen for A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(7):
            U[i_loc*2 + j_loc][c] = U2[r][c]
    for r in range(4):
        j_loc = r // 2
        k_loc = r % 2
        for c in range(7):
            V[j_loc*3 + k_loc][c] = V2[r][c]
    for r in range(4):
        k_loc = r // 2
        i_loc = r % 2
        for c in range(7):
            W[k_loc*2 + i_loc][c] = W2[r][c]
    
    # Place 2x2x1 schoolbook for A[0:2,0:2]*B[0:2,2:3]->C[0:2,2:3]
    for r in range(4):
        i_loc = r // 2
        j_loc = r % 2
        for c in range(4):
            U[i_loc*2 + j_loc][7+c] = U221[r][c]
    for r in range(6):
        j_loc = r // 2
        k_loc = r % 2  # k_loc is 0
        for c in range(4):
            V[j_loc*3 + 2][7+c] = V221[r][c]
    for r in range(4):
        k_loc = 0
        i_loc = r // 2
        for c in range(4):
            W[0*2 + i_loc][7+c] = W221[r][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    try:
        # Strategy: 
        # A is 4x5, B is 5x6, C is 4x6
        # Split: m=5 as 4+1, p=6 as 4+2
        # 
        # C[0:4, 0:4] = A[0:4,0:4]*B[0:4,0:4] + A[0:4,4:5]*B[4:5,0:4]
        # C[0:4, 4:6] = A[0:4,0:4]*B[0:4,4:6] + A[0:4,4:5]*B[4:5,4:6]
        #
        # Decompositions:
        # 1. 4x4x4 Strassen: rank 49 (A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4])
        # 2. 4x4x2 Strassen: rank 28 (A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6])
        # 3. 4x1x4 Schoolbook: rank 16 (A[0:4,4:5] * B[4:5,0:4] -> C[0:4,0:4])
        # 4. 4x1x2 Schoolbook: rank 8 (A[0:4,4:5] * B[4:5,4:6] -> C[0:4,4:6])
        #
        # Total: 49 + 28 + 16 + 8 = 101
        
        U44, V44, W44 = _build_strassen_4x4()
        U442, V442, W442 = _build_strassen_4x4x2()
        U414, V414, W414 = _schoolbook(4, 1, 4)
        U412, V412, W412 = _schoolbook(4, 1, 2)
        
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        total_rank = 49 + 28 + 16 + 8
        U = [[0]*total_rank for _ in range(n*m)]  # 4*5=20 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 5*6=30 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 6*4=24 rows
        
        # Place decomp1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
        offset = 0
        _decomp_block(4, 4, 4, 49, U44, V44, W44, 0, 0, 0, n, m, p, U, V, W, offset)
        
        # Place decomp2: A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]
        offset = 49
        _decomp_block(4, 4, 2, 28, U442, V442, W442, 0, 0, 4, n, m, p, U, V, W, offset)
        
        # Place decomp3: A[0:4,4:5] * B[4:5,0:4] -> C[0:4,0:4]
        offset = 77
        _decomp_block(4, 1, 4, 16, U414, V414, W414, 0, 4, 0, n, m, p, U, V, W, offset)
        
        # Place decomp4: A[0:4,4:5] * B[4:5,4:6] -> C[0:4,4:6]
        offset = 93
        _decomp_block(4, 1, 2, 8, U412, V412, W412, 0, 4, 4, n, m, p, U, V, W, offset)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
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
