import time
import numpy as np

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1
    T_float = T.astype(float)
    
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_check = np.einsum('ar,br,cr->abc', U_arr.T, V_arr.T, W_arr.T)
        return np.allclose(T_check, T_float, atol=1e-10)
    
    def get_baseline_66():
        U = [[0] * 66 for _ in range(n * m)]
        V = [[0] * 66 for _ in range(m * p)]
        W = [[0] * 66 for _ in range(p * n)]
        col = 0
        
        def add_schoolbook(i, j, k):
            nonlocal col
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
        
        def add_strassen_222(i_base, j_base, k_base):
            nonlocal col
            a_idx = i_base * m + j_base
            b_idx = i_base * m + (j_base + 1)
            c_idx = (i_base + 1) * m + j_base
            d_idx = (i_base + 1) * m + (j_base + 1)
            e_idx = j_base * p + k_base
            f_idx = j_base * p + (k_base + 1)
            g_idx = (j_base + 1) * p + k_base
            h_idx = (j_base + 1) * p + (k_base + 1)
            w00 = k_base * n + i_base
            w01 = (k_base + 1) * n + i_base
            w10 = k_base * n + (i_base + 1)
            w11 = (k_base + 1) * n + (i_base + 1)
            
            U[a_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1
            U[c_idx][col] = 1; U[d_idx][col] = 1
            V[e_idx][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1
            U[a_idx][col] = 1
            V[f_idx][col] = 1; V[h_idx][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1
            U[d_idx][col] = 1
            V[g_idx][col] = 1; V[e_idx][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1
            U[a_idx][col] = 1; U[b_idx][col] = 1
            V[h_idx][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1
            U[c_idx][col] = 1; U[a_idx][col] = -1
            V[e_idx][col] = 1; V[f_idx][col] = 1
            W[w11][col] = 1
            col += 1
            U[b_idx][col] = 1; U[d_idx][col] = -1
            V[g_idx][col] = 1; V[h_idx][col] = 1
            W[w00][col] = 1
            col += 1
        
        for j in range(4):
            for k in range(6):
                add_schoolbook(0, j, k)
        
        for j_block in range(2):
            for k_block in range(3):
                j_base = j_block * 2
                k_base = k_block * 2
                add_strassen_222(1, j_base, k_base)
        
        return U, V, W
    
    U_base, V_base, W_base = get_baseline_66()
    best_rank = 66
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    
    # Quick search for rank-65
    if time.time() < deadline - 5:
        U_arr = np.array(U_base, dtype=float)
        V_arr = np.array(V_base, dtype=float)
        W_arr = np.array(W_base, dtype=float)
        
        for elim in range(66):
            if time.time() > deadline - 5:
                break
            mask = [i for i in range(66) if i != elim]
            U_sub = U_arr[:, mask]
            V_sub = V_arr[:, mask]
            W_sub = W_arr[:, mask]
            T_sub = np.einsum('ar,br,cr->abc', U_sub.T, V_sub.T, W_sub.T)
            residual = T_float - T_sub
            if np.allclose(residual, 0, atol=1e-10):
                U_new = [[0]*65 for _ in range(n*m)]
                V_new = [[0]*65 for _ in range(m*p)]
                W_new = [[0]*65 for _ in range(p*n)]
                for r, orig in enumerate(mask):
                    for a in range(n*m): U_new[a][r] = U_arr[a, orig]
                    for b in range(m*p): V_new[b][r] = V_arr[b, orig]
                    for c in range(p*n): W_new[c][r] = W_arr[c, orig]
                if verify(U_new, V_new, W_new):
                    best_rank = 65
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    break
    
    return best_result

