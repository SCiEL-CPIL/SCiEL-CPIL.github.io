import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try to find rank 9 decomposition for 2x3x3
    # Then use it to build rank 27 for 2x3x9
    
    result_small = search_rank_9_2x3x3(deadline - 10)
    
    if result_small is not None:
        return construct_2x3x9_from_2x3x3(result_small)
    
    # Fallback to schoolbook
    return construct_schoolbook(n, m, p)

def search_rank_9_2x3x3(time_budget):
    n, m, p = 2, 3, 3
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def calc_error(U, V, W):
        Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = Res - T
        return np.sum(diff**2)

    random.seed(42)
    np.random.seed(42)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start = time.time()
    deadline = start + time_budget
    
    target_rank = 9
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize with random integers or half-integers
        U = np.random.randint(-1, 2, size=(num_u, target_rank)).astype(float)
        V = np.random.randint(-1, 2, size=(num_v, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(num_w, target_rank)).astype(float)
        
        current_err = calc_error(U, V, W)
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-9:
                return snap_and_verify_small(best_U, best_V, best_W, calc_error)
        
        # Local Search
        deltas = [0.5, -0.5, 1.0, -1.0]
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows = num_u
            elif mat_idx == 1:
                mat = V
                rows = num_v
            else:
                mat = W
                rows = num_w
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, target_rank - 1)
            
            old_val = mat[row, col]
            delta = random.choice(deltas)
            new_val = old_val + delta
            
            mat[row, col] = new_val
            new_err = calc_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                no_improve_count = 0
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-9:
                        return snap_and_verify_small(best_U, best_V, best_W, calc_error)
            else:
                mat[row, col] = old_val
                no_improve_count += 1

    if best_err < 1e-6:
        return snap_and_verify_small(best_U, best_V, best_W, calc_error)

    return None

def snap_and_verify_small(U, V, W, calc_error):
    bU = np.round(U * 2) / 2
    bV = np.round(V * 2) / 2
    bW = np.round(W * 2) / 2
    snap_err = calc_error(bU, bV, bW)
    if snap_err < 1e-9:
        return {
            "U": bU.tolist(),
            "V": bV.tolist(),
            "W": bW.tolist()
        }
    return None

def construct_2x3x9_from_2x3x3(small_decomp):
    n, m, p = 2, 3, 9
    n_s, m_s, p_s = 2, 3, 3
    rank_s = len(small_decomp['U'][0])
    
    U_s = np.array(small_decomp['U'])
    V_s = np.array(small_decomp['V'])
    W_s = np.array(small_decomp['W'])
    
    # Construct large matrices
    # U is (n*m) x (3*rank_s)
    # V is (m*p) x (3*rank_s)
    # W is (p*n) x (3*rank_s)
    
    U_large = np.zeros((n*m, 3*rank_s))
    V_large = np.zeros((m*p, 3*rank_s))
    W_large = np.zeros((p*n, 3*rank_s))
    
    for q in range(3):
        # U is the same for all blocks?
        # In the decomposition C_q = A B_q, U depends on A.
        # A is the same for all blocks.
        # So U_large[:, q*rank_s:(q+1)*rank_s] = U_s
        U_large[:, q*rank_s:(q+1)*rank_s] = U_s
        
        # V_large: row index j*p + k.
        # For block q, k is 0..2.
        # V_s row index j*p_s + k = j*3 + k.
        # V_large row index j*p + k_large = j*9 + (3q+k).
        for j in range(m):
            for k in range(p_s):
                idx_v_large = j*p + (3*q + k)
                idx_v_small = j*p_s + k
                V_large[idx_v_large, q*rank_s:(q+1)*rank_s] = V_s[idx_v_small, :]
                
        # W_large: row index k*n + i.
        # W_s row index k*n_s + i = k*2 + i.
        # W_large row index k_large*n + i = (3q+k)*2 + i.
        for k in range(p_s):
            for i in range(n):
                idx_w_large = (3*q + k)*n + i
                idx_w_small = k*n_s + i
                W_large[idx_w_large, q*rank_s:(q+1)*rank_s] = W_s[idx_w_small, :]
                
    return {
        "U": U_large.tolist(),
        "V": V_large.tolist(),
        "W": W_large.tolist()
    }

def construct_schoolbook(n, m, p):
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
