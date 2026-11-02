import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Rank 50 decomposition (Strassen-based)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
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

    def verify_fast(U, V, W):
        """Fast verification using numpy"""
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

    def try_rank(r, max_trials=100):
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Strategy: Structured sparse V and W based on block structure
            V = np.zeros((m_rows, r), dtype=np.int64)
            W = np.zeros((p_rows, r), dtype=np.int64)
            
            # Assign each column to a "type" based on which block it serves
            # Types: 0-6 for Strassen 2x2x2, 7-10 for last col, 11+ for A_R*B_B
            for c in range(r):
                # V: assign to 2-4 rows in specific blocks
                # Block 0: rows 0-8 (A_L * B_T, k=0..8, i=0)
                # Block 1: rows 9-17 (A_L * B_T, k=0..8, i=1)
                # Block 2: rows 18-26 (A_R * B_B)
                block_choice = random.random()
                if block_choice < 0.4:
                    # Strassen-type: rows in blocks 0 and 1
                    v_count = random.choice([3, 4])
                    v_indices = list(np.random.choice(18, size=min(v_count, 18), replace=False))
                    # Ensure at least one from each of rows 0-8 and 9-17 sometimes
                    if len(v_indices) < 2:
                        v_indices.append(random.choice(range(9)))
                        if len(v_indices) < 2:
                            v_indices.append(random.choice(range(9, 18)))
                elif block_choice < 0.7:
                    # Last col type
                    v_count = random.choice([2, 3])
                    v_indices = list(np.random.choice(18, size=min(v_count, 18), replace=False))
                else:
                    # A_R * B_B type
                    v_count = random.choice([1, 2])
                    v_indices = list(np.random.choice(9, size=min(v_count, 9), replace=False) + 18)
                
                for idx in v_indices:
                    V[idx, c] = random.choice([-1, 1])
                
                # W: assign to 1-3 rows
                # Rows: k*n+i for k in 0..8, i in 0..1
                # Group by k: rows k*2, k*2+1
                w_block_choice = random.random()
                if w_block_choice < 0.5:
                    # Strassen-type: pick 2-3 k values
                    w_count = random.choice([2, 3])
                    k_vals = np.random.choice(9, size=min(w_count, 9), replace=False)
                    w_indices = []
                    for k in k_vals:
                        w_indices.append(k * n + random.choice([0, 1]))
                elif w_block_choice < 0.8:
                    # Last col
                    w_count = random.choice([1, 2])
                    k_vals = np.random.choice(9, size=min(w_count, 9), replace=False)
                    w_indices = []
                    for k in k_vals:
                        w_indices.append(k * n + random.choice([0, 1]))
                else:
                    # A_R * B_B type
                    w_count = random.choice([1, 2])
                    k_vals = np.random.choice(9, size=min(w_count, 9), replace=False)
                    w_indices = []
                    for k in k_vals:
                        w_indices.append(k * n + random.choice([0, 1]))
                
                for idx in w_indices:
                    W[idx, c] = random.choice([-1, 1])
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r), dtype=np.float64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M @ u = 1 (least squares)
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integer
                    u_int = np.round(u).astype(np.int64)
                    
                    # Check exactness
                    if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                        valid = False
                        break
                        
                    U[i*m+j, :] = u_int
                    
            if not valid:
                continue
            
            # Full verification
            if verify_fast(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
        return None

    # Try to find a lower rank solution
    best_result = None
    # Try ranks 49 down to 46 with more trials for higher ranks
    for r in range(49, 45, -1):
        if time.time() > deadline - 25:
            break
        # More trials for ranks closer to 50
        trials = max(50, 300 - (49 - r) * 40)
        res = try_rank(r, max_trials=trials)
        if res:
            best_result = res
            break
        
    if best_result:
        return best_result
    
    # Fallback to rank 50
    return current_solution()
