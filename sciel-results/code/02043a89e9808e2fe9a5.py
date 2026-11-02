import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    rank_school = n * m * p  # 48
    
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    np.random.seed(42)
    for target_rank in range(46, rank_school):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.zeros((n*m, target_rank))
            V_rand = np.zeros((m*p, target_rank))
            W_rand = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                U_rand[i*m+j, r] = np.random.choice([-1, 1])
                V_rand[j*p+k, r] = np.random.choice([-1, 1])
                W_rand[k*n+i, r] = np.random.choice([-1, 1])
            
            if verify(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break
        
        if best_rank < rank_school:
            break
    
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
