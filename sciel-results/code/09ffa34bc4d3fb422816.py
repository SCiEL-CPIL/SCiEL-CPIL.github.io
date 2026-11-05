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

    # Try to find a better decomposition using local search over F_7
    field_size = 7
    field_elements = list(range(field_size))
    
    def tensor_error(U, V, W, rank):
        mismatches = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    s = s % field_size
                    if s != 1:
                        mismatches += 1
        return mismatches

    def verify_solution(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if s % field_size != 1:
                        return False
        return True

    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Try ranks from 50 to best_rank - 1
    for R in range(50, best_rank):
        if time.time() - start_time > max_time:
            break
            
        # Multiple restarts for each rank
        found = False
        for restart in range(20):
            if time.time() - start_time > max_time:
                break
                
            # Initialize random decomposition
            U = [[random.choice(field_elements) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice(field_elements) for _ in range(R)] for _ in range(m * p)]
            W = [[random.choice(field_elements) for _ in range(R)] for _ in range(p * n)]
            
            # Local search
            current_err = tensor_error(U, V, W, R)
            no_improve_count = 0
            max_iter = 5000
            
            while current_err > 0 and no_improve_count < 100 and time.time() - start_time < max_time:
                # Pick a random matrix and entry to perturb
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    old_val = U[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    U[idx][col] = new_val
                    new_err = tensor_error(U, V, W, R)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve_count = 0
                    else:
                        U[idx][col] = old_val
                        no_improve_count += 1
                elif mat == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    old_val = V[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    V[idx][col] = new_val
                    new_err = tensor_error(U, V, W, R)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve_count = 0
                    else:
                        V[idx][col] = old_val
                        no_improve_count += 1
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    old_val = W[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    W[idx][col] = new_val
                    new_err = tensor_error(U, V, W, R)
                    if new_err < current_err:
                        current_err = new_err
                        no_improve_count = 0
                    else:
                        W[idx][col] = old_val
                        no_improve_count += 1
                
                if current_err == 0:
                    break
            
            if current_err == 0:
                # Verify the solution
                if verify_solution(U, V, W, R):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = R
                    found = True
                    break
        
        if found:
            break

    return best_solution
