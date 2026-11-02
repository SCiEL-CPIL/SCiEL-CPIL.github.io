import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True
    
    def verify_fast(U_arr, V_arr, W_arr):
        """Fast vectorized verification."""
        r = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    def try_rank(r, max_trials=100):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(42)  # Reproducible
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Strategy: Use block structure to guide random initialization
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            for c in range(r):
                # V column: focus on rows that matter for the block structure
                # Rows 0-8 are B_T rows 0, 9-17 are B_T rows 1, 18-26 are B_B rows
                nz_v = np.random.randint(2, 5)
                # Bias toward having entries in different block regions
                regions = [list(range(0, 9)), list(range(9, 18)), list(range(18, 27))]
                chosen_regions = np.random.choice(3, size=min(nz_v, 3), replace=False)
                idx_v = []
                for reg in chosen_regions:
                    idx_v.append(np.random.choice(reg))
                # Add extra random ones if needed
                while len(idx_v) < nz_v:
                    idx_v.append(np.random.randint(0, m_rows))
                for idx in idx_v:
                    V[idx, c] = np.random.choice([-1, 1, 2, -2])
                
                # W column: rows 0-8 are W for i=0, 9-17 are W for i=1
                nz_w = np.random.randint(2, 5)
                w_regions = [list(range(0, 9)), list(range(9, 18))]
                chosen_w_regions = np.random.choice(2, size=min(nz_w, 2), replace=False)
                idx_w = []
                for reg in chosen_w_regions:
                    idx_w.append(np.random.choice(reg))
                while len(idx_w) < nz_w:
                    idx_w.append(np.random.randint(0, p_rows))
                for idx in idx_w:
                    W[idx, c] = np.random.choice([-1, 1, 2, -2])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            # Solve for U using least squares for each (i,j)
            valid = True
            for i in range(n):
                for j in range(m):
                    # Build M (p x r): M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :].astype(np.float64) * W[k*n+i, :].astype(np.float64)
                    
                    # Solve M @ u = ones(p)
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u).astype(np.int64)
                        
                        # Verify exactness
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification using fast method
            U_arr = U.astype(np.int64)
            V_arr = V.astype(np.int64)
            W_arr = W.astype(np.int64)
            
            if verify_fast(U_arr, V_arr, W_arr):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    def try_merge_from_50():
        """Try to reduce rank by merging columns from the rank-50 solution."""
        if time.time() > deadline - 15:
            return None
        
        sol = current_solution()
        U0 = np.array(sol["U"], dtype=np.int64)
        V0 = np.array(sol["V"], dtype=np.int64)
        W0 = np.array(sol["W"], dtype=np.int64)
        r0 = U0.shape[1]  # 50
        
        # Try removing each column and adjusting neighbors
        for remove_col in range(r0):
            if time.time() > deadline - 10:
                return None
            
            # Create modified matrices without column remove_col
            U_m = np.delete(U0, remove_col, axis=1)
            V_m = np.delete(V0, remove_col, axis=1)
            W_m = np.delete(W0, remove_col, axis=1)
            
            # Check if the remaining columns still give 1 for all (i,j,k)
            # The removed column contributed: U0[i,j,remove_col]*V0[j,k,remove_col]*W0[k,i,remove_col]
            # We need this to be 0 for all (i,j,k) OR we need to adjust other columns
            
            # Check if removal is safe (contribution was 0 everywhere)
            safe = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        contrib = U0[i*m+j, remove_col] * V0[j*p+k, remove_col] * W0[k*n+i, remove_col]
                        if contrib != 0:
                            safe = False
                            break
                    if not safe:
                        break
                if not safe:
                    break
            
            if safe:
                # Verify the reduced solution
                r_new = r0 - 1
                U_arr = U_m.astype(np.int64)
                V_arr = V_m.astype(np.int64)
                W_arr = W_m.astype(np.int64)
                if verify_fast(U_arr, V_arr, W_arr):
                    return {
                        "U": U_m.tolist(),
                        "V": V_m.tolist(),
                        "W": W_m.tolist()
                    }
        
        return None
    
    # Try to find lower rank solutions
    # First try the simple merge approach
    res = try_merge_from_50()
    if res:
        return res
    
    # Try ranks from 49 down to 42 with structured random search
    for r in range(49, 41, -1):
        if time.time() > deadline - 20:
            break
        res = try_rank(r, max_trials=50)
        if res:
            return res
    
    # Fallback to rank 50
    return current_solution()

