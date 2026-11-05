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
    field_elements = list(range(field_size))
    
    # Precompute indices for faster error calculation
    # We need to check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 mod 7
    # for all i in [0, n), j in [0, m), k in [0, p)
    
    def calculate_error(U, V, W, R):
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0
                    for c in range(R):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if s % field_size != 1:
                        errors += 1
        return errors

    start_time = time.time()
    max_time = 115  # Leave 5s buffer

    # Try ranks from 28 (CW-inspired lower bound guess) up to best_rank - 1
    # The directive suggests rank 28 is achievable.
    for R in range(28, best_rank):
        if time.time() - start_time > max_time:
            break
        
        found = False
        # Multiple random restarts for each rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                break
                
            # Initialize
            U = [[random.choice(field_elements) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice(field_elements) for _ in range(R)] for _ in range(m * p)]
            W = [[random.choice(field_elements) for _ in range(R)] for _ in range(p * n)]
            
            current_err = calculate_error(U, V, W, R)
            
            # Simulated Annealing / Local Search
            temperature = 10.0
            cooling = 0.95
            no_improve = 0
            max_iter_no_improve = 200
            
            while current_err > 0 and no_improve < max_iter_no_improve and time.time() - start_time < max_time:
                # Pick a random variable to change
                mat_id = random.randint(0, 2)
                if mat_id == 0:
                    idx = random.randint(0, n * m - 1)
                    col = random.randint(0, R - 1)
                    old_val = U[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    U[idx][col] = new_val
                    new_err = calculate_error(U, V, W, R)
                    delta = new_err - current_err
                    if delta < 0 or (temperature > 1e-6 and random.random() < np.exp(-delta / temperature)):
                        current_err = new_err
                        no_improve = 0 if delta < 0 else no_improve + 1
                    else:
                        U[idx][col] = old_val
                        no_improve += 1
                elif mat_id == 1:
                    idx = random.randint(0, m * p - 1)
                    col = random.randint(0, R - 1)
                    old_val = V[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    V[idx][col] = new_val
                    new_err = calculate_error(U, V, W, R)
                    delta = new_err - current_err
                    if delta < 0 or (temperature > 1e-6 and random.random() < np.exp(-delta / temperature)):
                        current_err = new_err
                        no_improve = 0 if delta < 0 else no_improve + 1
                    else:
                        V[idx][col] = old_val
                        no_improve += 1
                else:
                    idx = random.randint(0, p * n - 1)
                    col = random.randint(0, R - 1)
                    old_val = W[idx][col]
                    new_val = random.choice(field_elements)
                    if new_val == old_val:
                        continue
                    W[idx][col] = new_val
                    new_err = calculate_error(U, V, W, R)
                    delta = new_err - current_err
                    if delta < 0 or (temperature > 1e-6 and random.random() < np.exp(-delta / temperature)):
                        current_err = new_err
                        no_improve = 0 if delta < 0 else no_improve + 1
                    else:
                        W[idx][col] = old_val
                        no_improve += 1
                        
                temperature *= cooling
                
            if current_err == 0:
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = R
                found = True
                break # Found minimal rank for this R, stop searching higher R? No, we iterate R upwards.
                # Since we start from 28 and go up, the first one found is the minimal.
        
        if found:
            break

    return best_solution
