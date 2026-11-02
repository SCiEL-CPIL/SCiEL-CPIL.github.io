import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def build_decomposition_from_terms(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t, (u_list, v_list, w_list) in enumerate(terms):
            for r in range(n * m):
                U[r][t] = u_list[r]
            for r in range(m * p):
                V[r][t] = v_list[r]
            for r in range(p * n):
                W[r][t] = w_list[r]
        return U, V, W, rank
    
    def check_decomposition(U_r, V_r, W_r, rank):
        U_arr = np.array(U_r, dtype=np.float64)
        V_arr = np.array(V_r, dtype=np.float64)
        W_arr = np.array(W_r, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[u_idx(i,j)] * V_arr[v_idx(j,k)] * W_arr[w_idx(k,i)]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    # Build the structured Strassen/Schoolbook decomposition (rank 39)
    terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_list[idx] = val
        for idx, val in v_dict.items():
            v_list[idx] = val
        for idx, val in w_dict.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def strassen_block(col_offset):
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1},
            {w_idx(col_offset,1): 1, w_idx(col_offset+1,1): -1}
        )
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,col_offset+1): 1, v_idx(1,col_offset+1): -1},
            {w_idx(col_offset+1,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,col_offset): 1, v_idx(0,col_offset): -1},
            {w_idx(col_offset,0): 1, w_idx(col_offset,1): 1}
        )
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): -1, w_idx(col_offset+1,0): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,col_offset): 1, v_idx(0,col_offset+1): 1},
            {w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1}
        )
    
    def schoolbook_2x1(col_k):
        add_term({u_idx(0,0): 1}, {v_idx(0,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(0,1): 1}, {v_idx(1,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(1,0): 1}, {v_idx(0,col_k): 1}, {w_idx(col_k,1): 1})
        add_term({u_idx(1,1): 1}, {v_idx(1,col_k): 1}, {w_idx(col_k,1): 1})
    
    strassen_block(0)
    schoolbook_2x1(2)
    strassen_block(3)
    strassen_block(5)
    
    for col_k in range(7):
        add_term({u_idx(0,2): 1}, {v_idx(2,col_k): 1}, {w_idx(col_k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,col_k): 1}, {w_idx(col_k,1): 1})
    
    U, V, W, rank = build_decomposition_from_terms(terms)
    
    if not check_decomposition(U, V, W, rank):
        trivial_rank = n * m * p
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        rank = trivial_rank
    
    best_rank = rank
    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    
    # Try to reduce rank using optimization
    if time.time() < deadline - 30:
        def try_reduce_rank(target_rank, max_iter=2000, tol=1e-8):
            if time.time() > deadline - 10:
                return None
            
            num_u = n * m * target_rank
            num_v = m * p * target_rank
            num_w = p * n * target_rank
            
            U_init = np.array(U[:target_rank], dtype=np.float64).flatten()
            V_init = np.array(V[:target_rank], dtype=np.float64).flatten()
            W_init = np.array(W[:target_rank], dtype=np.float64).flatten()
            
            x0 = np.concatenate([U_init, V_init, W_init])
            
            def objective(x):
                U_flat = x[:num_u].reshape(n * m, target_rank)
                V_flat = x[num_u:num_u+num_v].reshape(m * p, target_rank)
                W_flat = x[num_u+num_v:].reshape(p * n, target_rank)
                
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = U_flat[u_idx(i,j)] * V_flat[v_idx(j,k)] * W_flat[w_idx(k,i)]
                            residual += (np.sum(prod) - 1.0) ** 2
                return residual
            
            init_res = objective(x0)
            if init_res < tol:
                U_out = x0[:num_u].reshape(n * m, target_rank)
                V_out = x0[num_u:num_u+num_v].reshape(m * p, target_rank)
                W_out = x0[num_u+num_v:].reshape(p * n, target_rank)
                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            
            try:
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    options={'maxiter': max_iter, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                return None
            
            if result.fun < tol:
                x_opt = result.x
                U_out = x_opt[:num_u].reshape(n * m, target_rank)
                V_out = x_opt[num_u:num_u+num_v].reshape(m * p, target_rank)
                W_out = x_opt[num_u+num_v:].reshape(p * n, target_rank)
                
                U_list = U_out.tolist()
                V_list = V_out.tolist()
                W_list = W_out.tolist()
                if check_decomposition(U_list, V_list, W_list, target_rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            return None
        
        for target_rank in range(rank - 1, max(30, rank - 5), -1):
            if time.time() > deadline - 10:
                break
            
            result = try_reduce_rank(target_rank, max_iter=2000)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
    
    return best_result
