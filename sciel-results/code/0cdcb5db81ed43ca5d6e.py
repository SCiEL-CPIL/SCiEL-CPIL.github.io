"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Split the 4x4 * 4x6 problem into 4x4 * 4x4 (first 4 columns of B) + 4x4 * 4x2 (last 2 columns of B)
    - For 4x4 * 4x4: use recursive Strassen (2 levels) giving rank 49
    - For 4x4 * 4x2: use Strassen on 2x2 blocks, giving rank 28
    - Total rank: 77
    - Then attempt to reduce rank by removing linearly dependent columns
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term to the decomposition."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        """
        u1 = {(oi, oj): 1, (oi+1, oj+1): 1}
        v1 = {(oj, ok): 1, (oj+1, ok+1): 1}
        w1 = {(oi, ok): 1, (oi+1, ok+1): 1}
        add_term(u1, v1, w1)
        
        u2 = {(oi+1, oj): 1, (oi+1, oj+1): 1}
        v2 = {(oj, ok): 1}
        w2 = {(oi+1, ok): 1, (oi+1, ok+1): -1}
        add_term(u2, v2, w2)
        
        u3 = {(oi, oj): 1}
        v3 = {(oj, ok+1): 1, (oj+1, ok+1): -1}
        w3 = {(oi, ok+1): 1, (oi+1, ok+1): 1}
        add_term(u3, v3, w3)
        
        u4 = {(oi+1, oj+1): 1}
        v4 = {(oj+1, ok): 1, (oj, ok): -1}
        w4 = {(oi, ok): 1, (oi+1, ok): 1}
        add_term(u4, v4, w4)
        
        u5 = {(oi, oj): 1, (oi, oj+1): 1}
        v5 = {(oj+1, ok+1): 1}
        w5 = {(oi, ok): -1, (oi, ok+1): 1}
        add_term(u5, v5, w5)
        
        u6 = {(oi+1, oj): 1, (oi, oj): -1}
        v6 = {(oj, ok): 1, (oj, ok+1): 1}
        w6 = {(oi+1, ok+1): 1}
        add_term(u6, v6, w6)
        
        u7 = {(oi, oj+1): 1, (oi+1, oj+1): -1}
        v7 = {(oj+1, ok): 1, (oj+1, ok+1): 1}
        w7 = {(oi, ok): 1}
        add_term(u7, v7, w7)

    def add_super_term(u_blocks, v_blocks, w_blocks, oi, oj, ok):
        """
        Add a Strassen super-term which is a sum of block products.
        """
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for u_t, v_t, w_t in terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (br, bc), coeff_u in u_blocks.items():
                for (r, c), coeff_t in u_t.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    u_col[i * m + j] += coeff_u * coeff_t
            
            for (br, bc), coeff_v in v_blocks.items():
                for (r, c), coeff_t in v_t.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    v_col[j * p + k] += coeff_v * coeff_t
            
            for (br, bc), coeff_w in w_blocks.items():
                for (r, c), coeff_t in w_t.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    w_col[k * n + i] += coeff_w * coeff_t
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def strassen_4x4(oi, oj, ok):
        """
        Add Strassen decomposition for 4x4 * 4x4 block multiplication.
        """
        add_super_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, oi, oj, ok)
        add_super_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, oi, oj, ok)
        add_super_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}, oi, oj, ok)
        add_super_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}, oi, oj, ok)
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    if _time_remaining() > 5.0:
        strassen_4x4(0, 0, 0)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    for rb in range(2):
        if _time_remaining() < 5.0:
            break
        strassen_2x2(rb*2, 0, 4)
        strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Attempt to reduce rank by finding linearly dependent columns
    if _time_remaining() > 30.0:
        U_final, V_final, W_final, rank = try_reduce_rank(U_final, V_final, W_final, n, m, p, rank)
    
    return {"U": U_final, "V": V_final, "W": W_final}

def try_reduce_rank(U, V, W, n, m, p, current_rank):
    """
    Try to reduce rank by identifying and removing linearly dependent columns.
    Returns (U, V, W, new_rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify(U_a, V_a, W_a):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_a[i*m+j, :] * V_a[j*p+k, :] * W_a[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U_arr, V_arr, W_arr):
        return U, V, W, current_rank
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = current_rank
    
    # Try removing columns one at a time from the end
    for remove_idx in range(current_rank - 1, 0, -1):
        if _time_remaining() < 15.0:
            break
        
        U_test = np.delete(best_U, remove_idx, axis=1)
        V_test = np.delete(best_V, remove_idx, axis=1)
        W_test = np.delete(best_W, remove_idx, axis=1)
        
        if verify(U_test, V_test, W_test):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = remove_idx
    
    if best_rank < current_rank:
        U_out = [[best_U[i][r] for r in range(best_rank)] for i in range(len(U))]
        V_out = [[best_V[i][r] for r in range(best_rank)] for i in range(len(V))]
        W_out = [[best_W[i][r] for r in range(best_rank)] for i in range(len(W))]
        return U_out, V_out, W_out, best_rank
    
    return U, V, W, current_rank
