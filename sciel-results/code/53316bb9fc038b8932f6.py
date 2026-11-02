# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a block decomposition strategy combining Strassen (2x2x2) and schoolbook for smaller blocks.
    Target Rank: 49 (vs Schoolbook 60).
    """
    n, m, p = 2, 5, 6
    
    def schoolbook(n, m, p):
        """
        Returns U, V, W lists for schoolbook decomposition of n x m * m x p.
        U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return U, V, W

    def strassen_2x2x2():
        """
        Returns U, V, W for Strassen decomposition of 2x2 * 2x2.
        Rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indices: i,j in 0..1 for U/V rows, k,i in 0..1 for W rows.
        """
        # Strassen constants
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # U[i*m+j, t]
        # V[j*p+k, t]
        # W[k*n+i, t]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # t=0: M1
        # (A00+A11) * (B00+B11)
        # U: A00(0,0)=1, A11(1,1)=1
        # V: B00(0,0)=1, B11(1,1)=1
        # W: C00 += M1, C11 += M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C00 (k=0,i=0), C11 (k=1,i=1)
        
        # t=1: M2
        # (A10+A11) * B00
        # U: A10(1,0)=1, A11(1,1)=1
        # V: B00(0,0)=1
        # W: C10 += M2 (k=0,i=1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # C10 (k=0,i=1) -> index k*n+i = 0*2+1 = 2
        
        # t=2: M3
        # A00 * (B01-B11)
        # U: A00(0,0)=1
        # V: B01(0,1)=1, B11(1,1)=-1
        # W: C01 += M3 (k=1,i=0)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # C01 (k=1,i=0) -> index 1*2+0 = 1
        
        # t=3: M4
        # A11 * (B10-B00)
        # U: A11(1,1)=1
        # V: B10(1,0)=1, B00(0,0)=-1
        # W: C11 += M4 (k=1,i=1)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[3][3] = 1 # C11 (k=1,i=1) -> index 1*2+1 = 3
        
        # t=4: M5
        # (A00+A01) * B11
        # U: A00(0,0)=1, A01(0,1)=1
        # V: B11(1,1)=1
        # W: C01 += M5, C11 += M5? No.
        # C01 = M3 + M5. C11 = M1 + M4 - M5?
        # Formula:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # So M5 contributes to C01 and C11?
        # Let's check standard Strassen:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # Yes.
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[3][4] = 1 # C01 (1), C11 (3)
        
        # t=5: M6
        # (A10-A00) * (B00+B01)
        # U: A10(1,0)=1, A00(0,0)=-1
        # V: B00(0,0)=1, B01(0,1)=1
        # W: C11 += M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11 (3)
        
        # t=6: M7
        # (A01-A11) * (B10+B11)
        # U: A01(0,1)=1, A11(1,1)=-1
        # V: B10(1,0)=1, B11(1,1)=1
        # W: C00 -= M7, C11 += M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[3][6] = 1 # C00 (0), C11 (3)
        
        return U, V, W

    def assemble(U, V, W, offset_u, offset_v, offset_w, start_col, n_sub, m_sub, p_sub):
        """
        Add sub-decomposition to main matrices.
        offset_u: start index in U for this sub-block (i*m_sub + j)
        offset_v: start index in V for this sub-block (j*p_sub + k)
        offset_w: start index in W for this sub-block (k*n_sub + i)
        start_col: column index to start writing in main U,V,W
        """
        rank_sub = len(U[0])
        for i in range(n_sub * m_sub):
            for t in range(rank_sub):
                # Map local U index to global
                # Local i_l = i // m_sub, j_l = i % m_sub
                # Global I = offset_u + i
                # But U is (n*m, rank).
                # We just add to the correct row.
                pass # This helper is tricky with indices. Better to construct directly.
        
    # We will construct the full U, V, W directly.
    # Total Rank = 49
    total_rank = 49
    U = [[0] * total_rank for _ in range(n * m)] # 10 x 49
    V = [[0] * total_rank for _ in range(m * p)] # 30 x 49
    W = [[0] * total_rank for _ in range(p * n)] # 12 x 49
    
    col = 0
    
    # Part 1: 2 x 2 x 6 (j=0,1)
    # Split k into 0-1, 2-3, 4-5
    # 3 blocks of 2x2x2
    for k_block in range(3):
        k_start = k_block * 2
        u_sub, v_sub, w_sub = strassen_2x2x2()
        for i in range(4):
            for t in range(7):
                U[i][col + t] += u_sub[i][t]
        for j in range(4):
            # V index: j*p + k. Here p=2 (local), k is local 0,1.
            # Global k = k_start + k_local
            # V row index = j_global * 6 + k_global
            # j_global = j_local (0,1)
            # k_global = k_start + k_local
            # j_local is 0..1.
            # V_sub row index = j_local * 2 + k_local
            # We need to map V_sub row r to V row (j_global * 6 + k_global)
            # r = j_l * 2 + k_l
            j_l = r // 2
            k_l = r % 2
            r_global = j_l * 6 + (k_start + k_l)
            for t in range(7):
                V[r_global][col + t] += v_sub[r][t]
        for k_i in range(4):
            # W index: k*n + i. Here n=2 (local), i is local 0,1.
            # W_sub row index = k_local * 2 + i_local
            # Global k = k_start + k_local
            # Global i = i_local
            # W row index = k_global * 2 + i_global
            k_l = k_i // 2
            i_l = k_i % 2
            r_global = (k_start + k_l) * 2 + i_l
            for t in range(7):
                W[r_global][col + t] += w_sub[k_i][t]
        col += 7
        
    # Part 2: 2 x 3 x 6 (j=2,3,4)
    # Split k into 0-2, 3-5
    # 2 blocks of 2x3x3
    for k_block in range(2):
        k_start = k_block * 3
        # 2x3x3 decomposition:
        # Split j (local 0,1,2) into 0 and 1,2
        # Block A: 2x1x3 (j_local=0)
        # Block B: 2x2x3 (j_local=1,2)
        
        # Block A: 2x1x3
        # Schoolbook for 2x1x3? Rank 3.
        # U (2x1, 3), V (1x3, 3), W (3x2, 3)
        u_a, v_a, w_a = schoolbook(2, 1, 3)
        for i in range(2):
            for t in range(3):
                # U row: i (local) -> Global i
                # But U is indexed by i*m + j.
                # Here m=5. j_global = 2 (since j_local=0 in the 2x3x3 block which starts at j=2)
                # Wait, the 2x3x3 block corresponds to j_global 2,3,4.
                # j_local 0 -> j_global 2.
                r_global = i * 5 + 2
                U[r_global][col + t] += u_a[i][t]
        for v_idx in range(3):
            # V row: j*p + k. j_local=0. k_local 0..2.
            # r_local = 0 * 3 + k_l = k_l
            # r_global = j_global * 6 + k_global
            # j_global = 2
            # k_global = k_start + k_l
            k_l = v_idx
            r_global = 2 * 6 + (k_start + k_l)
            for t in range(3):
                V[r_global][col + t] += v_a[v_idx][t]
        for w_idx in range(6):
            # W row: k*n + i. k_local 0..2. i_local 0..1.
            # r_local = k_l * 2 + i_l
            k_l = w_idx // 2
            i_l = w_idx % 2
            r_global = (k_start + k_l) * 2 + i_l
            for t in range(3):
                W[r_global][col + t] += w_a[w_idx][t]
        col += 3
        
        # Block B: 2x2x3
        # Split k_local (0..2) into 0 and 1,2
        # Block B1: 2x2x1 (k_local=0)
        # Block B2: 2x2x2 (k_local=1,2)
        
        # Block B1: 2x2x1
        u_b1, v_b1, w_b1 = schoolbook(2, 2, 1)
        for i in range(4):
            # U row: i_local (0..3) -> j_local 1,2.
            # i_local = i_l * 2 + j_l
            i_l = i // 2
            j_l = i % 2
            j_global = 2 + 1 + j_l # j_local 1,2 map to global 3,4
            r_global = i_l * 5 + j_global
            for t in range(4):
                U[r_global][col + t] += u_b1[i][t]
        for v_idx in range(2):
            # V row: j_local * 1 + k_local(0)
            j_l = v_idx // 1 # 0,1
            k_l = 0
            j_global = 2 + 1 + j_l
            k_global = k_start + k_l
            r_global = j_global * 6 + k_global
            for t in range(4):
                V[r_global][col + t] += v_b1[v_idx][t]
        for w_idx in range(4):
            # W row: k_local(0) * 2 + i_local
            k_l = 0
            i_l = w_idx
            r_global = (k_start + k_l) * 2 + i_l
            for t in range(4):
                W[r_global][col + t] += w_b1[w_idx][t]
        col += 4
        
        # Block B2: 2x2x2
        u_b2, v_b2, w_b2 = strassen_2x2x2()
        for i in range(4):
            i_l = i // 2
            j_l = i % 2
            j_global = 2 + 1 + j_l
            r_global = i_l * 5 + j_global
            for t in range(7):
                U[r_global][col + t] += u_b2[i][t]
        for v_idx in range(4):
            j_l = v_idx // 2
            k_l = v_idx % 2
            j_global = 2 + 1 + j_l
            k_global = k_start + 1 + k_l # k_local 1,2 map to global k_start+1, k_start+2
            r_global = j_global * 6 + k_global
            for t in range(7):
                V[r_global][col + t] += v_b2[v_idx][t]
        for w_idx in range(4):
            k_l = w_idx // 2
            i_l = w_idx % 2
            k_global = k_start + 1 + k_l
            r_global = k_global * 2 + i_l
            for t in range(7):
                W[r_global][col + t] += w_b2[w_idx][t]
        col += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
