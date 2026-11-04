import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # Index 1: i*m + j
    # Index 2: j*p + k
    # Index 3: k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    def local_search(U, V, W, max_iter=500):
        """Local search with half-integer constraint."""
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        
        # Compute initial cost efficiently
        def compute_cost(U_arr, V_arr, W_arr):
            recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
            return np.linalg.norm(T - recon) ** 2
        
        current_cost = compute_cost(current_U, current_V, current_W)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random matrix and a random entry
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = current_U
            elif mat_idx == 1:
                mat = current_V
            else:
                mat = current_W
            
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, mat.shape[1] - 1)
            
            old_val = mat[row, col]
            
            # Try perturbation: +/- 0.5 or +/- 1.0
            delta = random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            # Apply perturbation
            if mat_idx == 0:
                current_U[row, col] = new_val
            elif mat_idx == 1:
                current_V[row, col] = new_val
            else:
                current_W[row, col] = new_val
            
            new_cost = compute_cost(current_U, current_V, current_W)
            
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                # Revert
                if mat_idx == 0:
                    current_U[row, col] = old_val
                elif mat_idx == 1:
                    current_V[row, col] = old_val
                else:
                    current_W[row, col] = old_val
        
        return current_U, current_V, current_W, current_cost

    # Strategy: Try rank 41 by removing one schoolbook term at a time
    # For each of the 42 schoolbook terms, try removing it and optimizing
    target_rank = best_rank - 1
    found_better = False
    
    # Try each possible removed term
    for remove_col in range(n * m * p):
        if time.time() > deadline - 10:
            break
        
        # Create initial guess: schoolbook with column remove_col zeroed out
        # and redistribute its contribution to other columns
        U_init = U_sb.copy()
        V_init = V_sb.copy()
        W_init = W_sb.copy()
        
        # Zero out the removed column
        U_init[:, remove_col] = 0
        V_init[:, remove_col] = 0
        W_init[:, remove_col] = 0
        
        # Try local search from this initialization
        U_try, V_try, W_try, cost = local_search(U_init, V_init, W_init, max_iter=300)
        
        if cost < 1e-9:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
            found_better = True
            break
    
    # If rank 41 not found, try random initializations for rank 41
    if not found_better and time.time() < deadline - 10:
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            U_rand = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
            V_rand = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
            W_rand = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
            
            U_try, V_try, W_try, cost = local_search(U_rand, V_rand, W_rand, max_iter=200)
            
            if cost < 1e-9:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                found_better = True
                break
    
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
