import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Precompute indices for vectorized error calculation
    u_indices = np.array([(i * m + j) for i in range(n) for j in range(m) for k in range(p)])
    v_indices = np.array([(j * p + k) for i in range(n) for j in range(m) for k in range(p)])
    w_indices = np.array([(k * n + i) for i in range(n) for j in range(m) for k in range(p)])
    num_terms = n * m * p

    def error(U, V, W):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # U[u_indices, c] * V[v_indices, c] * W[w_indices, c]
        # Sum over c
        # Shape: (num_terms, rank)
        term_u = U_arr[u_indices]
        term_v = V_arr[v_indices]
        term_w = W_arr[w_indices]
        
        prod = term_u * term_v * term_w
        sums = prod.sum(axis=1)
        
        err = np.sum((sums - 1.0) ** 2)
        return err

    def random_init(rank, max_val=2):
        U = [[random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]) for _ in range(rank)] for _ in range(dim_u)]
        V = [[random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]) for _ in range(rank)] for _ in range(dim_v)]
        W = [[random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]) for _ in range(rank)] for _ in range(dim_w)]
        return U, V, W

    def mutate(U, V, W, rank):
        # Pick a random matrix and a random entry
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            mat = U
        elif mat_idx == 1:
            mat = V
        else:
            mat = W
            
        row = random.randint(0, len(mat) - 1)
        col = random.randint(0, rank - 1)
        
        # Change value by a small step
        steps = [-1, -0.5, 0.5, 1]
        new_val = mat[row][col] + random.choice(steps)
        # Clamp to reasonable range
        if new_val < -5: new_val = -5
        if new_val > 5: new_val = 5
        # Snap to half-integer
        new_val = round(new_val * 2) / 2
        mat[row][col] = new_val

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    best_rank = 24
    best_solution = None
    
    # Try to find a solution with rank < 24
    # Start from 23 down to 18
    for r in range(23, 17, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = random_init(r)
            curr_err = error(U, V, W)
            
            # Local search
            temp = 1.0
            for step in range(1000):
                if time.time() - start_time > time_limit:
                    break
                    
                if curr_err < 1e-9:
                    best_rank = r
                    best_solution = (U, V, W)
                    break
                    
                U_new = [row[:] for row in U]
                V_new = [row[:] for row in V]
                W_new = [row[:] for row in W]
                
                mutate(U_new, V_new, W_new, r)
                
                new_err = error(U_new, V_new, W_new)
                
                if new_err < curr_err or random.random() < np.exp((curr_err - new_err) / temp):
                    U, V, W = U_new, V_new, W_new
                    curr_err = new_err
                
                temp *= 0.995 # Cooling
                
            if best_solution is not None:
                break
                
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = 24
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
