import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Rank 36 Strassen-based decomposition for 2x4x5
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)
    
    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        # Strassen patterns for 2x2 * 2x2
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        strassen_patterns = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # M4
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # M6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            
            col_offset += 1
        
        return col_offset
    
    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        # Schoolbook for 2x2 * 2x1
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            
            col_offset += 1
        
        return col_offset
    
    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    
    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"
    
    # Verify decomposition
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True
    
    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
