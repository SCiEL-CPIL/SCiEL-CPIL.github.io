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
    
    # Try ranks from a lower bound upward
    for r in range(36, best_rank):
        if time.time() > deadline:
            break
        
        # Try many random attempts with diverse entry sets
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Mix of integer and half-integer entries
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
            
            if verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best
