# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1;U[3][0]=1;V[0][0]=1;V[3][0]=1;W[0][0]=1;W[3][0]=1
        U[2][1]=1;U[3][1]=1;V[0][1]=1;W[2][1]=1;W[3][1]=-1
        U[0][2]=1;V[1][2]=1;V[3][2]=-1;W[1][2]=1;W[3][2]=1
        U[3][3]=1;V[2][3]=1;V[0][3]=-1;W[0][3]=1;W[2][3]=1
        U[0][4]=1;U[1][4]=1;V[3][4]=1;W[0][4]=-1;W[1][4]=1
        U[2][5]=1;U[0][5]=-1;V[0][5]=1;V[1][5]=1;W[3][5]=1
        U[1][6]=1;U[3][6]=-1;V[2][6]=1;V[3][6]=1;W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c]: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c]: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c]: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c]: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c]: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c]: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6_strassen():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        
        # Pre-compute non-zero indices for sub-matrices to speed up assembly
        u_nz = [(r, c, U_sub[r][c]) for r in range(4) for c in range(rank_sub) if U_sub[r][c] != 0]
        v_nz = [(r, c, V_sub[r][c]) for r in range(6) for c in range(rank_sub) if V_sub[r][c] != 0]
        w_nz = [(r, c, W_sub[r][c]) for r in range(6) for c in range(rank_sub) if W_sub[r][c] != 0]

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for u_local, c, u_val in u_nz:
                i_l, j_l = u_local // 2, u_local % 2
                for block, coeff in term['A']:
                    ro, co = u_offset[block]
                    U[(i_l+ro)*m + (j_l+co)][base_col+c] += u_val*coeff
            for v_local, c, v_val in v_nz:
                j_l, k_l = v_local // 3, v_local % 3
                for block, coeff in term['B']:
                    ro, co = v_offset[block]
                    V[(j_l+ro)*p + (k_l+co)][base_col+c] += v_val*coeff
            for w_local, c, w_val in w_nz:
                k_l, i_l = w_local // 2, w_local % 2
                for block, coeff in term['C']:
                    ro, co = w_offset[block]
                    W[(k_l+ro)*n + (i_l+co)][base_col+c] += w_val*coeff
        return U, V, W

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Primary: Strassen-based rank 77
    U, V, W = build_4x4x6_strassen()
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
