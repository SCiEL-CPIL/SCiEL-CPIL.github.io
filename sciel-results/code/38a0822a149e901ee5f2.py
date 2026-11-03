import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Discrete values allowed
    DISCRETE_VALS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    def compute_residual(U, V, W):
        # Returns the residual tensor
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return prod - T

    def count_errors(residual, tol=1e-6):
        # Count number of entries that are not close to 0
        # Note: The target T has 0s and 1s.
        # The decomposition must produce exactly T.
        # So residual must be 0 everywhere.
        return np.sum(np.abs(residual) > tol)

    def discrete_hill_climb(U, V, W, R, max_steps=2000, temp_init=1.0, temp_min=0.01, cooling=0.99):
        # U, V, W are numpy arrays of shape (dim, R) with values in DISCRETE_VALS
        # We use Simulated Annealing to escape local minima
        
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        
        current_res = compute_residual(current_U, current_V, current_W)
        current_err = count_errors(current_res)
        
        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
        best_err = current_err
        
        temp = temp_init
        
        rng = np.random.RandomState(42)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            if current_err == 0:
                break
                
            # Pick a random variable and index to change
            # Total variables = U_dim*R + V_dim*R + W_dim*R
            total_vars = U_dim * R + V_dim * R + W_dim * R
            var_idx = rng.randint(0, total_vars)
            
            if var_idx < U_dim * R:
                u_idx, r_idx = np.unravel_index(var_idx, (U_dim, R))
                # Change U[u_idx, r_idx]
                old_val = current_U[u_idx, r_idx]
                # Pick a new random value from DISCRETE_VALS
                new_val = DISCRETE_VALS[rng.randint(0, len(DISCRETE_VALS))]
                if new_val == old_val:
                    continue
                current_U[u_idx, r_idx] = new_val
                
                new_res = compute_residual(current_U, current_V, current_W)
                new_err = count_errors(new_res)
                
                # Acceptance criterion (Metropolis)
                delta_err = new_err - current_err
                if delta_err <= 0 or rng.random() < np.exp(-delta_err / temp):
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = current_U.copy()
                        best_V = current_V.copy()
                        best_W = current_W.copy()
                else:
                    # Revert
                    current_U[u_idx, r_idx] = old_val
                    
            elif var_idx < U_dim * R + V_dim * R:
                local_idx = var_idx - U_dim * R
                v_idx, r_idx = np.unravel_index(local_idx, (V_dim, R))
                old_val = current_V[v_idx, r_idx]
                new_val = DISCRETE_VALS[rng.randint(0, len(DISCRETE_VALS))]
                if new_val == old_val:
                    continue
                current_V[v_idx, r_idx] = new_val
                
                new_res = compute_residual(current_U, current_V, current_W)
                new_err = count_errors(new_res)
                
                delta_err = new_err - current_err
                if delta_err <= 0 or rng.random() < np.exp(-delta_err / temp):
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = current_U.copy()
                        best_V = current_V.copy()
                        best_W = current_W.copy()
                else:
                    current_V[v_idx, r_idx] = old_val
                    
            else:
                local_idx = var_idx - (U_dim * R + V_dim * R)
                w_idx, r_idx = np.unravel_index(local_idx, (W_dim, R))
                old_val = current_W[w_idx, r_idx]
                new_val = DISCRETE_VALS[rng.randint(0, len(DISCRETE_VALS))]
                if new_val == old_val:
                    continue
                current_W[w_idx, r_idx] = new_val
                
                new_res = compute_residual(current_U, current_V, current_W)
                new_err = count_errors(new_res)
                
                delta_err = new_err - current_err
                if delta_err <= 0 or rng.random() < np.exp(-delta_err / temp):
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = current_U.copy()
                        best_V = current_V.copy()
                        best_W = current_W.copy()
                else:
                    current_W[w_idx, r_idx] = old_val
                    
            temp *= cooling
            if temp < temp_min:
                temp = temp_min
                
        return best_err == 0, best_U, best_V, best_W

    def random_discrete_init(U_dim, V_dim, W_dim, R, seed_val):
        rng = np.random.RandomState(seed_val)
        U = rng.choice(DISCRETE_VALS, size=(U_dim, R))
        V = rng.choice(DISCRETE_VALS, size=(V_dim, R))
        W = rng.choice(DISCRETE_VALS, size=(W_dim, R))
        return U, V, W

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 45 to 49
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        R = target_rank
        found = False
        
        # More restarts for lower ranks as they are harder
        num_restarts = 10 if target_rank <= 46 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            seed_val = 100 + restart * 1000 + R
            U, V, W = random_discrete_init(U_dim, V_dim, W_dim, R, seed_val)
            
            # Run discrete hill climbing
            # Max steps needs to be enough to converge but fast enough.
            # 1000-2000 steps is reasonable.
            success, U_sol, V_sol, W_sol = discrete_hill_climb(U, V, W, R, max_steps=1500)
            
            if success:
                best_solution = (U_sol, V_sol, W_sol)
                best_rank = R
                break
        
        if found or best_solution is not None and best_rank == R:
             break

    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()
