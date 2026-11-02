import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def verify_443(U, V, W):
    # n=4, m=4, p=3
    n, m, p = 4, 4, 3
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def search_443(deadline):
    n, m, p = 4, 4, 3
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
                
    for r in range(16, best_rank):
        if time.time() > deadline:
            break
        for attempt in range(100):
            if time.time() > deadline:
                break
            def rand_entry():
                choice = random.random()
                if choice < 0.3:
                    return random.choice([0, 1, -1, 2, -2])
                elif choice < 0.6:
                    return random.choice([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
                else:
                    return random.choice([0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5])
            
            U = [[rand_entry() for _ in range(r)] for _ in range(n*m)]
            V = [[rand_entry() for _ in range(r)] for _ in range(m*p)]
            W = [[rand_entry() for _ in range(r)] for _ in range(p*n)]
            
            if verify_443(U, V, W):
                return {"U": U, "V": V, "W": W, "rank": r}
        
    return {"U": best_U, "V": best_V, "W": best_W, "rank": best_rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook as default
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find a low rank decomposition for 4x4x3 and expand
    # This targets the block structure C = [A B1, A B2, A B3]
    sub_res = search_443(deadline)
    r_sub = sub_res["rank"]
    
    if r_sub < 48: # Only expand if we found something better than schoolbook for 443
        total_rank = 3 * r_sub
        if total_rank < best_rank:
            U_full = [[0] * total_rank for _ in range(n*m)]
            V_full = [[0] * total_rank for _ in range(m*p)]
            W_full = [[0] * total_rank for _ in range(p*n)]
            
            U_sub = sub_res["U"]
            V_sub = sub_res["V"]
            W_sub = sub_res["W"]
            
            # U is shared across blocks
            for i in range(n):
                for j in range(m):
                    for r_local in range(r_sub):
                        for b in range(3):
                            r_full = b * r_sub + r_local
                            U_full[i*m+j][r_full] = U_sub[i*m+j][r_local]
            
            # V and W are block specific
            for b in range(3):
                for j in range(m):
                    for k_sub in range(3):
                        k_full = b * 3 + k_sub
                        for r_local in range(r_sub):
                            r_full = b * r_sub + r_local
                            V_full[j*p+k_full][r_full] = V_sub[j*3+k_sub][r_local]
                            
                for k_sub in range(3):
                    for i in range(n):
                        k_full = b * 3 + k_sub
                        for r_local in range(r_sub):
                            r_full = b * r_sub + r_local
                            W_full[k_full*n+i][r_full] = W_sub[k_sub*n+i][r_local]
            
            if verify(U_full, V_full, W_full, n, m, p):
                best = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = total_rank
    
    return best
