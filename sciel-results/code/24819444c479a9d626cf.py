import numpy as np
import random
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    field_size = 7
    
    start_time = time.time()
    max_time = 115
    
    # Precompute index mappings
    u_idxs = []
    v_idxs = []
    w_idxs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
    
    u_idxs = np.array(u_idxs)
    v_idxs = np.array(v_idxs)
    w_idxs = np.array(w_idxs)
    
    def tensor_error_np(U_np, V_np, W_np):
        U_sel = U_np[u_idxs, :]
        V_sel = V_np[v_idxs, :]
        W_sel = W_np[w_idxs, :]
        prod = U_sel * V_sel * W_sel
        sums = prod.sum(axis=1) % field_size
        return int(np.sum(sums != 1))
    
    def try_rank(R, num_restarts=10, max_iter=5000):
        for restart in range(num_restarts):
            if time.time() - start_time > max_time:
                return None
            
            U_np = np.random.randint(0, field_size, size=(n * m, R))
            V_np = np.random.randint(0, field_size, size=(m * p, R))
            W_np = np.random.randint(0, field_size, size=(p * n, R))
            
            current_err = tensor_error_np(U_np, V_np, W_np)
            
            if current_err == 0:
                return (U_np, V_np, W_np, R)
            
            no_improve = 0
            max_no_improve = 200
            
            for it in range(max_iter):
                if time.time() - start_time > max_time:
                    return None
                if current_err == 0:
                    return (U_np, V_np, W_np, R)
                if no_improve > max_no_improve:
                    break
                
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    old_val = U_np[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    U_np[idx, col] = new_val
                    new_err = tensor_error_np(U_np, V_np, W_np)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        U_np[idx, col] = old_val
                        no_improve += 1
                elif mat == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    old_val = V_np[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    V_np[idx, col] = new_val
                    new_err = tensor_error_np(U_np, V_np, W_np)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        V_np[idx, col] = old_val
                        no_improve += 1
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    old_val = W_np[idx, col]
                    new_val = random.randint(0, field_size - 1)
                    if new_val == old_val:
                        continue
                    W_np[idx, col] = new_val
                    new_err = tensor_error_np(U_np, V_np, W_np)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve = 0
                    else:
                        W_np[idx, col] = old_val
                        no_improve += 1
            
            if current_err == 0:
                return (U_np, V_np, W_np, R)
        
        return None
    
    for R in range(30, best_rank):
        if time.time() - start_time > max_time:
            break
        
        result = try_rank(R)
        if result is not None:
            U_np, V_np, W_np, R_found = result
            best_solution = {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
            best_rank = R_found
            break
    
    return best_solution
