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

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

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

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Uses Strassen (rank 7) for first 2 columns + schoolbook (rank 4) for 3rd column.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # Strassen part for columns 0-1 (rank 7)
        U_s, V_s, W_s = build_2x2x2_strassen()
        # V_s index: (j,k) -> j*2+k, so V_s[0]=B[0,0], V_s[1]=B[0,1], V_s[2]=B[1,0], V_s[3]=B[1,1]
        # W_s index: (k,i) -> k*2+i, so W_s[0]=C[0,0], W_s[1]=C[0,1], W_s[2]=C[1,0], W_s[3]=C[1,1]
        
        # For 2x2*2x3, V is 2x3 = 6 entries, W is 2x3 = 6 entries
        # V index: (j,k) -> j*3+k
        # W index: (k,i) -> k*2+i
        
        v_map = {0:0, 1:1, 2:3, 3:4}  # V_s[j*2+k] -> V[j*3+k] for k=0,1
        w_map = {0:0, 1:1, 2:2, 3:3}  # W_s[k*2+i] -> W[k*2+i]
        
        for c in range(7):
            for r in range(4):
                if U_s[r][c] != 0:
                    U[r][c] += U_s[r][c]
            for r in range(4):
                if V_s[r][c] != 0:
                    V[v_map[r]][c] += V_s[r][c]
            for r in range(4):
                if W_s[r][c] != 0:
                    W[w_map[r]][c] += W_s[r][c]
        
        # Schoolbook part for column 2 (rank 4)
        U_sb, V_sb, W_sb = build_2x2x1()
        base = 7
        v_sb_map = {0:2, 1:5}  # V_sb[j*1+k] -> V[j*3+2] for k=0
        w_sb_map = {0:4, 1:5}  # W_sb[k*2+i] -> W[k*2+i]
        
        for c in range(4):
            for r in range(4):
                if U_sb[r][c] != 0:
                    U[r][base+c] += U_sb[r][c]
            for r in range(2):
                if V_sb[r][c] != 0:
                    V[v_sb_map[r]][base+c] += V_sb[r][c]
            for r in range(2):
                if W_sb[r][c] != 0:
                    W[w_sb_map[r]][base+c] += W_sb[r][c]
        
        return U, V, W

    def build_4x4x6():
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
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
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

    # Attempt rank reduction by finding linearly dependent columns
    def try_reduce_rank(U, V, W, n, m, p, max_rank, time_budget):
        """Try to reduce rank by identifying and removing linearly dependent columns."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if current_rank <= max_rank:
            return U, V, W
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        # Try removing columns one at a time from the end
        for remove_idx in range(current_rank - 1, max_rank - 1, -1):
            if time.time() > time_budget:
                break
            
            # Remove column remove_idx
            U_test = np.delete(best_U, remove_idx, axis=1)
            V_test = np.delete(best_V, remove_idx, axis=1)
            W_test = np.delete(best_W, remove_idx, axis=1)
            
            # Check if the remaining columns still satisfy the constraint
            valid = True
            for i in range(n):
                if not valid: break
                for j in range(m):
                    if not valid: break
                    for k in range(p):
                        prod = U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :]
                        if abs(sum(prod) - 1.0) > 1e-6:
                            valid = False
                            break
            
            if valid:
                best_U = U_test
                best_V = V_test
                best_W = W_test
                best_rank -= 1
        
        if best_rank < current_rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return U, V, W

    # Attempt rank reduction
    if time.time() < deadline - 5:
        U, V, W = try_reduce_rank(U, V, W, n, m, p, 50, deadline - 5)

    # Final verification
    def verify_final(U, V, W, n, m, p):
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

    if not verify_final(U, V, W, n, m, p):
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
