import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find better decompositions
    for rank in range(25, rank_school):
        if time.time() > deadline - 30:
            break
            
        found = False
        max_iters = 50000
        iters = 0
        
        while iters < max_iters and not found and time.time() < deadline - 30:
            iters += 1
            
            # Random initialization with small integers
            scale = random.choice([1, 2, 3])
            U = np.random.randint(-scale, scale + 1, (n * m, rank))
            V = np.random.randint(-scale, scale + 1, (m * p, rank))
            W = np.random.randint(-scale, scale + 1, (p * n, rank))
            
            # Compute error
            def compute_error(U, V, W):
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                            err += (val - 1) ** 2
                return err
            
            err = compute_error(U, V, W)
            
            if err == 0:
                found = True
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
            
            # Local search
            for _ in range(100):
                if time.time() > deadline - 30:
                    break
                    
                # Pick a random matrix and entry to change
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = (random.randint(0, n*m-1), random.randint(0, rank-1))
                    old_val = U[idx]
                    U[idx] += random.choice([-2, -1, 1, 2])
                elif mat == 'V':
                    idx = (random.randint(0, m*p-1), random.randint(0, rank-1))
                    old_val = V[idx]
                    V[idx] += random.choice([-2, -1, 1, 2])
                else:
                    idx = (random.randint(0, p*n-1), random.randint(0, rank-1))
                    old_val = W[idx]
                    W[idx] += random.choice([-2, -1, 1, 2])
                
                new_err = compute_error(U, V, W)
                if new_err < err:
                    err = new_err
                    if err == 0:
                        found = True
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        break
                else:
                    if mat == 'U':
                        U[idx] = old_val
                    elif mat == 'V':
                        V[idx] = old_val
                    else:
                        W[idx] = old_val
        
        if found:
            break
    
    # Convert to lists
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
