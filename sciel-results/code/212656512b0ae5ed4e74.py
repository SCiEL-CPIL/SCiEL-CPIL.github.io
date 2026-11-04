import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    # Greedy peeling with multiple restarts
    def greedy_peel(T_tensor, max_rank=20, n_restarts=5):
        best_result = None
        best_rank = max_rank + 1
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            R = T_tensor.copy()
            U_list, V_list, W_list = [], [], []
            
            # Random initialization
            np.random.seed(restart * 12345 + 67890)
            
            for step in range(max_rank):
                if time.time() > deadline - 15:
                    break
                if np.max(np.abs(R)) < 1e-12:
                    break
                
                # Find best rank-1 approximation
                # Initialize
                if step == 0:
                    u = R.sum(axis=(1,2))
                    v = R.sum(axis=(0,2))
                    w = R.sum(axis=(0,1))
                else:
                    u = np.random.randn(nm)
                    v = np.random.randn(mp)
                    w = np.random.randn(pn)
                
                u = u / (np.linalg.norm(u) + 1e-15)
                v = v / (np.linalg.norm(v) + 1e-15)
                w = w / (np.linalg.norm(w) + 1e-15)
                
                # Power iteration
                for it in range(200):
                    vw = np.outer(v, w)
                    u_new = np.einsum('abc,bc->a', R, vw)
                    u_norm = np.linalg.norm(u_new)
                    if u_norm < 1e-15:
                        break
                    u = u_new / u_norm
                    
                    uw = np.outer(u, w)
                    v_new = np.einsum('abc,ac->b', R, uw)
                    v_norm = np.linalg.norm(v_new)
                    if v_norm < 1e-15:
                        break
                    v = v_new / v_norm
                    
                    uv = np.outer(u, v)
                    w_new = np.einsum('abc,ab->c', R, uv)
                    w_norm = np.linalg.norm(w_new)
                    if w_norm < 1e-15:
                        break
                    w = w_new / w_norm
                
                uvw_val = np.einsum('a,b,c,abc->', u, v, w, R)
                if abs(uvw_val) < 1e-12:
                    break
                
                # Scale: we want u⊗v⊗w to have the right magnitude
                # The contribution is uvw_val * u⊗v⊗w (since u,v,w are unit)
                # Actually, the rank-1 tensor is (uvw_val) * u⊗v⊗w
                # But we want to subtract it from R
                U_list.append(u * uvw_val)
                V_list.append(v)
                W_list.append(w)
                R -= np.einsum('a,b,c->abc', U_list[-1], V_list[-1], W_list[-1])
            
            if len(U_list) == 0:
                continue
            
            U_arr = np.array(U_list).T
            V_arr = np.array(V_list).T
            W_arr = np.array(W_list).T
            rank = len(U_list)
            err = verify_decomp(U_arr, V_arr, W_arr, rank)
            
            if err < best_err:
                best_err = err
                best_rank = rank
                best_result = (U_arr, V_arr, W_arr)
            
            if err < 1e-10:
                break
        
        return best_result, best_rank, best_err

    # Try greedy peeling
    result, rank_g, err_g = greedy_peel(T, max_rank=20, n_restarts=3)
    
    if result is not None and err_g < 1e-8:
        U_g, V_g, W_g = result
        # Try to round to half-integers
        U_r = np.round(U_g * 2) / 2.0
        V_r = np.round(V_g * 2) / 2.0
        W_r = np.round(W_g * 2) / 2.0
        
        if np.max(np.abs(U_r - U_g)) < 1e-6 and np.max(np.abs(V_r - V_g)) < 1e-6 and np.max(np.abs(W_r - W_g)) < 1e-6:
            err_r = verify_decomp(U_r, V_r, W_r, rank_g)
            if err_r < 1e-10:
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_g
            else:
                best_U, best_V, best_W, best_rank = U_g, V_g, W_g, rank_g
        else:
            best_U, best_V, best_W, best_rank = U_g, V_g, W_g, rank_g
    else:
        best_U, best_V, best_W, best_rank = None, None, None, 999

    # Fallback to Strassen rank-28
    if best_rank > 28 or best_U is None:
        rank = 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1
        
        X1 = [0, 1, 4, 5]
        Y1 = [0, 1, 4, 5]
        Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)
        X2 = [2, 3, 6, 7]
        Y2 = [8, 9, 12, 13]
        Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)
        X3 = [0, 1, 4, 5]
        Y3 = [2, 3, 6, 7]
        Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)
        X4 = [2, 3, 6, 7]
        Y4 = [10, 11, 14, 15]
        Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)
        
        best_U = np.array(U, dtype=float)
        best_V = np.array(V, dtype=float)
        best_W = np.array(W, dtype=float)
        best_rank = 28

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
