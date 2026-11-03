import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank accumulator
    total_rank = 0
    # Global matrices
    # We don't know the total rank beforehand easily without summing, 
    # so we'll collect blocks first.
    
    blocks_U = []
    blocks_V = []
    blocks_W = []
    
    def add_block(n_sub, m_sub, p_sub, u_sub, v_sub, w_sub, 
                  i_offset, j_offset, k_offset):
        """
        Add a sub-decomposition to the global lists.
        n_sub, m_sub, p_sub are dimensions of the sub-problem.
        u_sub: (n_sub * m_sub, rank_sub)
        v_sub: (m_sub * p_sub, rank_sub)
        w_sub: (p_sub * n_sub, rank_sub)
        offsets map local indices to global.
        """
        rank_sub = u_sub[0].__len__() if u_sub else 0
        if rank_sub == 0: return
        
        new_U = [[0]*rank_sub for _ in range(n*m)]
        new_V = [[0]*rank_sub for _ in range(m*p)]
        new_W = [[0]*rank_sub for _ in range(p*n)]
        
        for r in range(rank_sub):
            for i_local in range(n_sub):
                for j_local in range(m_sub):
                    val = u_sub[i_local*m_sub + j_local][r]
                    if val != 0:
                        i_global = i_offset + i_local
                        j_global = j_offset + j_local
                        new_U[i_global*m + j_global][r] = val
                        
            for j_local in range(m_sub):
                for k_local in range(p_sub):
                    val = v_sub[j_local*p_sub + k_local][r]
                    if val != 0:
                        j_global = j_offset + j_local
                        k_global = k_offset + k_local
                        new_V[j_global*p + k_global][r] = val
                        
            for k_local in range(p_sub):
                for i_local in range(n_sub):
                    val = w_sub[k_local*n_sub + i_local][r]
                    if val != 0:
                        k_global = k_offset + k_local
                        i_global = i_offset + i_local
                        new_W[k_global*n + i_global][r] = val
                        
        blocks_U.append(new_U)
        blocks_V.append(new_V)
        blocks_W.append(new_W)
        
    def get_schoolbook(n_s, m_s, p_s):
        """Returns U, V, W for schoolbook multiplication of n_s x m_s and m_s x p_s."""
        rank = n_s * m_s * p_s
        U = [[0]*rank for _ in range(n_s*m_s)]
        V = [[0]*rank for _ in range(m_s*p_s)]
        W = [[0]*rank for _ in range(p_s*n_s)]
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[i*m_s + j][col] = 1
                    V[j*p_s + k][col] = 1
                    W[k*n_s + i][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_2x2x2():
        """Returns U, V, W for Strassen 2x2x2."""
        # Standard Strassen decomposition
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Indices in U/V/W correspond to:
        # U: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # V: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1: (A11+A22)(B11+B22)
        # U: A11(0)=1, A22(3)=1
        # V: B11(0)=1, B22(3)=1
        # W: C11(0)=1, C22(3)=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A21+A22)B11
        # U: A21(2)=1, A22(3)=1
        # V: B11(0)=1
        # W: C21(2)=1, C22(3)=-1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3: A11(B12-B22)
        # U: A11(0)=1
        # V: B12(1)=1, B22(3)=-1
        # W: C12(1)=1, C22(3)=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: A22(B21-B11)
        # U: A22(3)=1
        # V: B21(2)=1, B11(0)=-1
        # W: C11(0)=1, C21(2)=1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A11+A12)B22
        # U: A11(0)=1, A12(1)=1
        # V: B22(3)=1
        # W: C11(0)=-1, C12(1)=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (A21-A11)(B11+B12)
        # U: A21(2)=1, A11(0)=-1
        # V: B11(0)=1, B12(1)=1
        # W: C22(3)=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A12-A22)(B21+B22)
        # U: A12(1)=1, A22(3)=-1
        # V: B21(2)=1, B22(3)=1
        # W: C11(0)=1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # --- Construction ---
    
    # 1. A1 (2x5) * B1 (5x2) -> C1 (2x2)
    # Split inner 5 into 2 (B1a) and 3 (B1b)
    # 1a: A1a (2x2) * B1a (2x2) -> Strassen
    U_s, V_s, W_s = get_strassen_2x2x2()
    add_block(2, 2, 2, U_s, V_s, W_s, 
              i_offset=0, j_offset=0, k_offset=0)
              
    # 1b: A1b (2x3) * B1b (3x2) -> Rank 9
    # Use split of inner 3 into 1+2? No, A1b is 2x3, B1b is 3x2.
    # Split inner 3 (cols of A1b, rows of B1b) into 1+2.
    # Sub-1b-1: A1b_1 (2x1) * B1b_1 (1x2) -> Schoolbook Rank 2
    U_s1, V_s1, W_s1 = get_schoolbook(2, 1, 2)
    add_block(2, 1, 2, U_s1, V_s1, W_s1,
              i_offset=0, j_offset=2, k_offset=0) # A cols 2-2, B rows 2-2, C cols 0-1
              
    # Sub-1b-2: A1b_2 (2x2) * B1b_2 (2x2) -> Strassen Rank 7
    U_s2, V_s2, W_s2 = get_strassen_2x2x2()
    add_block(2, 2, 2, U_s2, V_s2, W_s2,
              i_offset=0, j_offset=3, k_offset=0) # A cols 3-4, B rows 3-4, C cols 0-1

    # 2. A1 (2x5) * B2 (5x3) -> C2 (2x3)
    # Split inner 5 into 2 (B2a) and 3 (B2b)
    # 2a: A1a (2x2) * B2a (2x3) -> Rank 11
    # Split output 3 into 1+2
    # Sub-2a-1: A1a (2x2) * B2a_1 (2x1) -> Schoolbook Rank 4
    U_s3, V_s3, W_s3 = get_schoolbook(2, 2, 1)
    add_block(2, 2, 1, U_s3, V_s3, W_s3,
              i_offset=0, j_offset=0, k_offset=2) # A cols 0-1, B rows 0-1, C col 2
              
    # Sub-2a-2: A1a (2x2) * B2a_2 (2x2) -> Strassen Rank 7
    U_s4, V_s4, W_s4 = get_strassen_2x2x2()
    add_block(2, 2, 2, U_s4, V_s4, W_s4,
              i_offset=0, j_offset=0, k_offset=3) # A cols 0-1, B rows 0-1, C cols 3-4

    # 2b: A1b (2x3) * B2b (3x3) -> Rank 15
    # Split output 3 into 1+2
    # Sub-2b-1: A1b (2x3) * B2b_1 (3x1) -> Schoolbook Rank 6
    U_s5, V_s5, W_s5 = get_schoolbook(2, 3, 1)
    add_block(2, 3, 1, U_s5, V_s5, W_s5,
              i_offset=0, j_offset=2, k_offset=2) # A cols 2-4, B rows 2-4, C col 2
              
    # Sub-2b-2: A1b (2x3) * B2b_2 (3x2) -> Rank 9
    # Split inner 3 into 1+2
    # Sub-2b-2-1: A1b_1 (2x1) * B2b_2_1 (1x2) -> Schoolbook Rank 2
    U_s6, V_s6, W_s6 = get_schoolbook(2, 1, 2)
    add_block(2, 1, 2, U_s6, V_s6, W_s6,
              i_offset=0, j_offset=2, k_offset=3) # A cols 2-2, B rows 2-2, C cols 3-4
              
    # Sub-2b-2-2: A1b_2 (2x2) * B2b_2_2 (2x2) -> Strassen Rank 7
    U_s7, V_s7, W_s7 = get_strassen_2x2x2()
    add_block(2, 2, 2, U_s7, V_s7, W_s7,
              i_offset=0, j_offset=3, k_offset=3) # A cols 3-4, B rows 3-4, C cols 3-4

    # 3. A2 (1x5) * B (5x5) -> C3 (1x5)
    # Schoolbook Rank 5
    U_s8, V_s8, W_s8 = get_schoolbook(1, 5, 5)
    add_block(1, 5, 5, U_s8, V_s8, W_s8,
              i_offset=2, j_offset=0, k_offset=0) # A row 2, B all, C row 2

    # Combine all blocks
    total_rank = sum(len(b[0]) for b in blocks_U)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for u_b, v_b, w_b in zip(blocks_U, blocks_V, blocks_W):
        r = len(u_b[0])
        for i in range(n*m):
            for c in range(r):
                U[i][col_offset + c] = u_b[i][c]
        for i in range(m*p):
            for c in range(r):
                V[i][col_offset + c] = v_b[i][c]
        for i in range(p*n):
            for c in range(r):
                W[i][col_offset + c] = w_b[i][c]
        col_offset += r
        
    return {"U": U, "V": V, "W": W}
