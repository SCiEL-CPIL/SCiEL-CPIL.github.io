import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    deadline = start_time + time_limit - safety_margin
    
    n, m, p = 2, 3, 9
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18

    # Target tensor indices
    # T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target_indices.append((a_idx, b_idx, c_idx))

    def verify_fast(U, V, W):
        """Faster verification using vectorization"""
        R = U.shape[1]
        for a_idx, b_idx, c_idx in target_indices:
            total = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
            if abs(total - 1.0) > 1e-6:
                return False
        return True

    def make_baseline():
        """Create the baseline rank-50 decomposition"""
        rank = 50
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        # 4 blocks of 2x2
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
            
            U[u_a, col] += 1
            U[u_d, col] += 1
            V[v_e, col] += 1
            V[v_h, col] += 1
            
            U[u_c, col+1] += 1
            U[u_d, col+1] += 1
            V[v_e, col+1] += 1
            
            U[u_a, col+2] += 1
            V[v_f, col+2] += 1
            V[v_h, col+2] -= 1
            
            U[u_d, col+3] += 1
            V[v_g, col+3] += 1
            V[v_e, col+3] -= 1
            
            U[u_a, col+4] += 1
            U[u_b, col+4] += 1
            V[v_h, col+4] += 1
            
            U[u_c, col+5] += 1
            U[u_a, col+5] -= 1
            V[v_e, col+5] += 1
            V[v_f, col+5] += 1
            
            U[u_b, col+6] += 1
            U[u_d, col+6] -= 1
            V[v_g, col+6] += 1
            V[v_h, col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00, col] += 1
            W[w_00, col+3] += 1
            W[w_00, col+4] -= 1
            W[w_00, col+6] += 1
            
            W[w_01, col+2] += 1
            W[w_01, col+4] += 1
            
            W[w_10, col+1] += 1
            W[w_10, col+3] += 1
            
            W[w_11, col] += 1
            W[w_11, col+1] -= 1
            W[w_11, col+2] += 1
            W[w_11, col+5] += 1
            
            col += 7
        
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a, col] += 1
        V[v_e, col] += 1
        U[u_b, col+1] += 1
        V[v_g, col+1] += 1
        U[u_c, col+2] += 1
        V[v_e, col+2] += 1
        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        
        W[w_08, col] += 1
        W[w_08, col+1] += 1
        W[w_18, col+2] += 1
        W[w_18, col+3] += 1
        
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0, col] += 1
            V[v_b, col] += 1
            W[k * n + 0, col] += 1
            U[u_ar1, col+1] += 1
            V[v_b, col+1] += 1
            W[k * n + 1, col+1] += 1
            col += 2
        
        return U, V, W

    U, V, W = make_baseline()
    best_rank = U.shape[1]
    
    if not verify_fast(U, V, W):
        pass

    def try_remove_column(U, V, W, col_idx):
        R = U.shape[1]
        if R <= 1:
            return None, None, None
        
        U_new = np.delete(U, col_idx, axis=1)
        V_new = np.delete(V, col_idx, axis=1)
        W_new = np.delete(W, col_idx, axis=1)
        
        if verify_fast(U_new, V_new, W_new):
            return U_new, V_new, W_new
        return None, None, None

    R = best_rank
    while True:
        if time.time() > deadline:
            break
        
        removed = False
        for col_idx in range(R):
            if time.time() > deadline:
                break
            
            U_new, V_new, W_new = try_remove_column(U, V, W, col_idx)
            if U_new is not None:
                U, V, W = U_new, V_new, W_new
                R = U.shape[1]
                best_rank = min(best_rank, R)
                removed = True
                break
        
        if not removed:
            break

    random.seed(42)
    best_found = best_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        R_cur = U_cur.shape[1]
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            if R_cur <= best_found:
                break
            
            col_idx = random.randint(0, R_cur - 1)
            U_test = np.delete(U_cur, col_idx, axis=1)
            V_test = np.delete(V_cur, col_idx, axis=1)
            W_test = np.delete(W_cur, col_idx, axis=1)
            
            if verify_fast(U_test, V_test, W_test):
                U_cur, V_cur, W_cur = U_test, V_test, W_test
                R_cur = U_cur.shape[1]
        
        if R_cur < best_found:
            best_found = R_cur
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    U, V, W = best_U, best_V, best_W
    
    def to_list_of_lists(matrix):
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(rounded - round(rounded)) < 1e-9:
                    rounded = round(rounded)
                row_list.append(rounded)
            result.append(row_list)
        return result
    
    U_list = to_list_of_lists(U)
    V_list = to_list_of_lists(V)
    W_list = to_list_of_lists(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
