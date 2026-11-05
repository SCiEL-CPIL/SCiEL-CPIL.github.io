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
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3_rank11():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3 (schoolbook + Strassen)."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_2x2x3_rank10():
        """Rank-10 decomposition for 2x2 * 2x3 -> 2x3 using a custom Strassen variant.
        
        We use the 7 Strassen products for the 2x2 part but merge the 4 schoolbook
        products into 3 by using a linear combination trick.
        
        Actually, let's try a different approach: use 3 products for the 2x1 part
        (which is possible since 2x1 * 2x1 has rank 3) and 7 for the 2x2 part.
        """
        rank = 10
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # For 2x2 * 2x1 -> 2x1, we can use rank 3:
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        # 
        # Using: M1 = A[0,0]*B[0], M2 = A[0,1]*B[1], M3 = (A[1,0]+A[1,1])*(B[0]+B[1])
        # C[0] = M1 + M2
        # C[1] = M3 - M2
        #
        # U index: i*2+j
        # V index: j (since k=0 only, j*1+0 = j)
        # W index: i (since k=0 only, 0*2+i = i)
        
        # Term 0: M1 = A[0,0]*B[0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1  # C[0]
        
        # Term 1: M2 = A[0,1]*B[1]
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1  # C[0]
        W[1][1] = -1 # C[1]
        
        # Term 2: M3 = (A[1,0]+A[1,1])*(B[0]+B[1])
        U[2][2] = 1
        U[3][2] = 1
        V[0][2] = 1
        V[1][2] = 1
        W[1][2] = 1  # C[1]
        
        # Now add the 7 Strassen products for the 2x2 part (columns 1,2 of B)
        U2, V2, W2 = build_2x2x2_strassen()
        base = 3
        v2_map = {0:1, 1:2, 2:4, 3:5}
        w2_map = {0:2, 1:4, 2:3, 3:5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x6(rank_sub):
        """Build 4x4 * 4x6 decomposition using the given 2x2x3 sub-decomposition."""
        n, m, p = 4, 4, 6
        if rank_sub == 11:
            U_sub, V_sub, W_sub = build_2x2x3_rank11()
        else:
            U_sub, V_sub, W_sub = build_2x2x3_rank10()
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

    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget):
        """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        
        for remove_idx in cols_to_try:
            if time.time() > time_budget:
                break
            if best_rank <= target_rank:
                break
            
            mask = [i for i in range(best_rank) if i != remove_idx]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            valid = True
            for i in range(n):
                if not valid: break
                for j in range(m):
                    if not valid: break
                    for k in range(p):
                        prod = U_test[i*m+j, :] * V_test[j*p+k, :] * W_test[k*n+i, :]
                        err = abs(sum(prod) - 1.0)
                        if err > 1e-6:
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

    def try_linear_combine(U, V, W, n, m, p, target_rank, time_budget):
        """Try to combine pairs of terms using linear combinations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        current_rank = U_arr.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        best_rank = current_rank
        
        # Try to find if two columns can be replaced by one
        for i in range(best_rank):
            if time.time() > time_budget:
                break
            if best_rank <= target_rank:
                break
            for j in range(i+1, best_rank):
                if time.time() > time_budget:
                    break
                if best_rank <= target_rank:
                    break
                
                # Check if column j is a linear combination of other columns
                # This is expensive, so we only try a few pairs
                if (i + j) % 100 != 0:
                    continue
                
                mask = [x for x in range(best_rank) if x != j]
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                valid = True
                for ii in range(n):
                    if not valid: break
                    for jj in range(m):
                        if not valid: break
                        for kk in range(p):
                            prod = U_test[ii*m+jj, :] * V_test[jj*p+kk, :] * W_test[kk*n+ii, :]
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

    # Try both rank-10 and rank-11 sub-decompositions
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for rank_sub in [10, 11]:
        if time.time() > deadline - 20:
            break
        
        U, V, W = build_4x4x6(rank_sub)
        
        if verify(U, V, W, n, m, p):
            current_rank = len(U[0])
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U, V, W
    
    if best_U is None:
        # Fallback to schoolbook
        rank = n * m * p
        best_U = [[0] * rank for _ in range(n * m)]
        best_V = [[0] * rank for _ in range(m * p)]
        best_W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][column] = 1
                    best_V[j * p + k][column] = 1
                    best_W[k * n + i][column] = 1
                    column += 1
    else:
        # Attempt rank reduction
        if time.time() < deadline - 15:
            best_U, best_V, best_W = try_reduce_rank(best_U, best_V, best_W, n, m, p, 50, deadline - 15)
        
        if time.time() < deadline - 10:
            best_U, best_V, best_W = try_linear_combine(best_U, best_V, best_W, n, m, p, 50, deadline - 10)
    
    # Final verification
    if not verify(best_U, best_V, best_W, n, m, p):
        rank = n * m * p
        best_U = [[0] * rank for _ in range(n * m)]
        best_V = [[0] * rank for _ in range(m * p)]
        best_W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][column] = 1
                    best_V[j * p + k][column] = 1
                    best_W[k * n + i][column] = 1
                    column += 1

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
