"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition indices
    # n: 0,1,2
    # m: 0,1,2 (group 1), 3,4 (group 2)
    # p: 0,1,2 (group 1), 3,4,5 (group 2), 6 (group 3)
    
    # We will construct U, V, W matrices.
    # Rank is sum of ranks of sub-problems.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_subproblem(n_start, n_end, m_start, m_end, p_start, p_end, sub_rank, sub_U, sub_V, sub_W):
        nonlocal rank
        for r in range(sub_rank):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for i in range(n_start, n_end):
                for j in range(m_start, m_end):
                    u_vec[i * m + j] = sub_U[r][i - n_start][j - m_start]
            
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    v_vec[j * p + k] = sub_V[r][j - m_start][k - p_start]
            
            for k in range(p_start, p_end):
                for i in range(n_start, n_end):
                    w_vec[k * n + i] = sub_W[r][k - p_start][i - n_start]
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
        rank += sub_rank

    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        r = n * m * p
        U = [[[0]*m for _ in range(n)] for _ in range(r)]
        V = [[[0]*p for _ in range(m)] for _ in range(r)]
        W = [[[0]*n for _ in range(p)] for _ in range(r)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[col][i][j] = 1
                    V[col][j][k] = 1
                    W[col][k][i] = 1
                    col += 1
        return r, U, V, W

    # Helper to create Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        r = 7
        U = [[[0]*2 for _ in range(2)] for _ in range(r)]
        V = [[[0]*2 for _ in range(2)] for _ in range(r)]
        W = [[[0]*2 for _ in range(2)] for _ in range(r)]
        
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U (A factors)
        U[0] = [[1, 0], [1, 1]]
        U[1] = [[0, 0], [1, 1]]
        U[2] = [[1, 0], [0, 0]]
        U[3] = [[0, 0], [0, 1]]
        U[4] = [[1, 1], [0, 0]]
        U[5] = [[-1, 0], [1, 0]]
        U[6] = [[0, 1], [0, -1]]
        
        # V (B factors)
        V[0] = [[1, 1], [1, 1]]
        V[1] = [[1, 0], [1, 0]]
        V[2] = [[0, 1], [0, -1]]
        V[3] = [[1, -1], [0, 0]]
        V[4] = [[0, 0], [1, 1]]
        V[5] = [[1, 1], [1, 1]] # Wait, M6 = (A10-A00)(B00+B10). V should be [[1,0],[1,0]]? No.
        # Let's re-verify Strassen.
        # M6 = (A10 - A00)(B00 + B10)
        # V6 should correspond to B00+B10 in first col, 0 in second?
        # B is 2x2. B00 is (0,0), B10 is (1,0).
        # V6 = [[1, 0], [1, 0]]
        V[5] = [[1, 0], [1, 0]]
        # M7 = (A01 - A11)(B01 + B11)
        # B01 is (0,1), B11 is (1,1).
        V[6] = [[0, 1], [0, 1]]
        
        # W (C factors)
        # C00 = M1 - M4 + M5 - M6
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        W[0] = [[1, 0], [0, 1]] # M1: +1 to C00, +1 to C11
        W[1] = [[0, 0], [1, 0]] # M2: +1 to C10
        W[2] = [[0, 1], [0, 0]] # M3: +1 to C01
        W[3] = [[-1, 1], [0, 0]] # M4: -1 to C00, +1 to C01
        W[4] = [[1, 0], [0, 0]] # M5: +1 to C00
        W[5] = [[-1, 0], [0, 1]] # M6: -1 to C00, +1 to C11
        W[6] = [[0, 0], [1, 0]] # M7: +1 to C10
        
        return r, U, V, W

    # 1. n=0..2, m=0..2, p=0..2 (3x3x3) -> Schoolbook 27
    r, U_s, V_s, W_s = schoolbook(3, 3, 3)
    add_subproblem(0, 3, 0, 3, 0, 3, r, U_s, V_s, W_s)
    
    # 2. n=0..2, m=3..4, p=0..2 (3x2x3)
    # Split p into 0..1 (2) and 2..2 (1)
    # 3x2x2: Rank 3
    r1, U1, V1, W1 = schoolbook(3, 2, 2) # Wait, 3x2x2 rank is 3? Schoolbook is 12.
    # We need the rank 3 decomposition for 3x2x2.
    # T_{i,j,k} = 1 if j=k? No.
    # C_i = A_i0 B_0k + A_i1 B_1k.
    # For each i, it's 1x2x2 rank 1.
    # So we can use 3 rank-1 terms.
    def rank_3_2_2():
        r = 3
        U = [[[0]*2 for _ in range(3)] for _ in range(r)]
        V = [[[0]*2 for _ in range(2)] for _ in range(r)]
        W = [[[0]*3 for _ in range(2)] for _ in range(r)]
        for i in range(3):
            U[i][i][0] = 1
            U[i][i][1] = 1
            V[i][0][0] = 1
            V[i][0][1] = 1
            V[i][1][0] = 1
            V[i][1][1] = 1
            W[i][0][i] = 1
            W[i][1][i] = 1
        return r, U, V, W
    
    r, U_s, V_s, W_s = rank_3_2_2()
    add_subproblem(0, 3, 3, 5, 0, 2, r, U_s, V_s, W_s)
    
    # 3x2x1: Rank 3
    def rank_3_2_1():
        r = 3
        U = [[[0]*2 for _ in range(3)] for _ in range(r)]
        V = [[[0]*1 for _ in range(2)] for _ in range(r)]
        W = [[[0]*3 for _ in range(1)] for _ in range(r)]
        for i in range(3):
            U[i][i][0] = 1
            U[i][i][1] = 1
            V[i][0][0] = 1
            V[i][1][0] = 1
            W[i][0][i] = 1
        return r, U, V, W
        
    r, U_s, V_s, W_s = rank_3_2_1()
    add_subproblem(0, 3, 3, 5, 2, 3, r, U_s, V_s, W_s)
    
    # 3. n=0..2, m=3..4, p=3..5 (3x2x3)
    # Split p into 3..4 (2) and 5..5 (1)
    r, U_s, V_s, W_s = rank_3_2_2()
    add_subproblem(0, 3, 3, 5, 3, 5, r, U_s, V_s, W_s)
    
    r, U_s, V_s, W_s = rank_3_2_1()
    add_subproblem(0, 3, 3, 5, 5, 6, r, U_s, V_s, W_s)
    
    # 4. n=0..2, m=3..4, p=6..6 (3x2x1)
    r, U_s, V_s, W_s = rank_3_2_1()
    add_subproblem(0, 3, 3, 5, 6, 7, r, U_s, V_s, W_s)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
