"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

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
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows

        # Term 0: A[0,0]*B[0] -> C[0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1

        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]

        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        # M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        # Part 1: 2x2 * 2x1 (cols 0 of B, col 0 of C)
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                U[r][c] += U1[r][c]
            for r in range(2):
                V[r*3][c] += V1[r][c]
                W[r][c] += W1[r][c]

        # Part 2: 2x2 * 2x2 (cols 1,2 of B, cols 1,2 of C)
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        # V map: local (j,k) -> global (j, k+1)
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global indices: 1:(0,1), 2:(0,2), 4:(1,1), 5:(1,2)
        v_map = {0: 1, 1: 2, 2: 4, 3: 5}
        # W map: local (k,i) -> global (k+1, i)
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global indices: 2:(1,0), 4:(2,0), 3:(1,1), 5:(2,1)
        w_map = {0: 2, 1: 4, 2: 3, 3: 5}
        
        for c in range(7):
            for r in range(4):
                U[r][base + c] += U2[r][c]
            for r in range(4):
                V[v_map[r]][base + c] += V2[r][c]
                W[w_map[r]][base + c] += W2[r][c]

        return U, V, W

    def build_4x4x6():
        """Rank-77 decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = 11
        total_rank = 7 * rank_sub

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        # Strassen terms for 4x4 * 4x6
        # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in block coords
        # U offsets (row, col) in block coords
        u_off = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        # V offsets (row, col) in block coords. Note V is 4x6, blocks are 2x3.
        v_off = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        # W offsets (row, col) in block coords. Note W is 6x4, blocks are 3x2.
        w_off = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}

        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                # U
                for u_loc in range(4):
                    val = U_sub[u_loc][c]
                    if val == 0: continue
                    i_l, j_l = divmod(u_loc, 2)
                    for b, coeff in term['A']:
                        br, bc = u_off[b]
                        gi, gj = i_l + br, j_l + bc
                        U[gi*m + gj][base_col + c] += val * coeff
                
                # V
                for v_loc in range(6):
                    val = V_sub[v_loc][c]
                    if val == 0: continue
                    j_l, k_l = divmod(v_loc, 3)
                    for b, coeff in term['B']:
                        br, bc = v_off[b]
                        gj, gk = j_l + br, k_l + bc
                        V[gj*p + gk][base_col + c] += val * coeff

                # W
                for w_loc in range(6):
                    val = W_sub[w_loc][c]
                    if val == 0: continue
                    k_l, i_l = divmod(w_loc, 2)
                    for b, coeff in term['C']:
                        br, bc = w_off[b]
                        gk, gi = k_l + br, i_l + bc
                        W[gk*n + gi][base_col + c] += val * coeff

        return U, V, W

    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
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
