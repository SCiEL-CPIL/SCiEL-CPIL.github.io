import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Search for low rank decomposition of 2x3x3
    n_s, m_s, p_s = 2, 3, 3
    n_s_rows = n_s * m_s
    m_s_rows = m_s * p_s
    p_s_rows = p_s * n_s

    def check_small(U, V, W, rank):
        # U: 6xR, V: 9xR, W: 6xR
        # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    a = i * m_s + j
                    b = j * p_s + k
                    c = k * n_s + i
                    val = 0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if val != 1:
                        return False
        return True

    def extend_to_full(U_s, V_s, W_s, rank):
        # U_full: 6xR
        # V_full: 27xR
        # W_full: 18xR
        U_full = np.zeros((n_rows, rank), dtype=np.int64)
        V_full = np.zeros((m_rows, rank), dtype=np.int64)
        W_full = np.zeros((p_rows, rank), dtype=np.int64)
        
        U_full[:] = U_s
        
        for j in range(m_s):
            for k in range(p_s):
                for block in range(3):
                    k_full = block * p_s + k
                    v_idx_full = j * p + k_full
                    v_idx_small = j * p_s + k
                    V_full[v_idx_full, :] = V_s[v_idx_small, :]
                    
                    w_idx_full = k_full * n_s + 0 # i is 0..1
                    # W index is k*n + i.
                    # For a given k_full, we need to fill W_full[k_full*n + i, :]
                    # W_s index is k*n_s + i.
                    # So W_full[k_full*n_s + i, :] = W_s[k*p_s + i, :] ? No.
                    # W_s is for p_s=3. W_s index is k_small * 2 + i.
                    # W_full index is k_full * 2 + i.
                    for i in range(n_s):
                        W_full[k_full * n_s + i, :] = W_s[k * n_s + i, :]
                        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    best_result = schoolbook()
    best_rank = 54

    # Try ranks for 2x3x3
    for rank in range(12, 18):
        if time.time() > deadline - 20:
            break
            
        max_trials = 200
        for t in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randint(-2, 3, size=(n_s_rows, rank))
            V = np.random.randint(-2, 3, size=(m_s_rows, rank))
            W = np.random.randint(-2, 3, size=(p_s_rows, rank))
            
            if check_small(U, V, W, rank):
                result = extend_to_full(U, V, W, rank)
                return result

    return best_result
# EVOLVE-BLOCK-END
