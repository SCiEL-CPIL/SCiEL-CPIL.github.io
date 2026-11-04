# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank

    def compute_error(U, V, W, n, m, p):
        """Compute max error of decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def is_valid(U, V, W, n, m, p, tol=1e-8):
        return compute_error(U, V, W, n, m, p) < tol

    # Discrete values to consider: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    discrete_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    # Try to find a lower rank decomposition using discrete local search
    # We will try ranks from 119 down to 80
    # For each rank, we do a few random restarts
    
    target_ranks = list(range(119, 79, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        n_restarts = 10
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if found:
                break
                
            np.random.seed(42 + restart * 1000 + target_rank)
            
            # Initialize with random discrete values
            # Bias towards small values
            U_try = np.zeros((n*m, target_rank))
            V_try = np.zeros((m*p, target_rank))
            W_try = np.zeros((p*n, target_rank))
            
            # Fill with random discrete values
            for c in range(target_rank):
                # Randomly select some rows to be non-zero
                # To keep it sparse, we can pick a few rows
                # But for now, let's just fill all with small random discrete values
                U_try[:, c] = np.random.choice(discrete_vals, size=n*m)
                V_try[:, c] = np.random.choice(discrete_vals, size=m*p)
                W_try[:, c] = np.random.choice(discrete_vals, size=p*n)
                
            # Local Search / Hill Climbing
            current_err = compute_error(U_try, V_try, W_try, n, m, p)
            best_err = current_err
            best_U_local = U_try.copy()
            best_V_local = V_try.copy()
            best_W_local = W_try.copy()
            
            # Iterate for a fixed number of steps
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                if current_err < 1e-8:
                    found = True
                    break
                    
                # Pick a random variable to change
                # We can change a value in U, V, or W
                mat_choice = np.random.choice(3) # 0: U, 1: V, 2: W
                if mat_choice == 0:
                    mat = U_try
                    dims = (n*m, target_rank)
                elif mat_choice == 1:
                    mat = V_try
                    dims = (m*p, target_rank)
                else:
                    mat = W_try
                    dims = (p*n, target_rank)
                    
                r_idx = np.random.randint(0, dims[0])
                c_idx = np.random.randint(0, dims[1])
                
                old_val = mat[r_idx, c_idx]
                new_val = np.random.choice(discrete_vals)
                
                if old_val == new_val:
                    continue
                    
                mat[r_idx, c_idx] = new_val
                
                new_err = compute_error(U_try, V_try, W_try, n, m, p)
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U_local = U_try.copy()
                        best_V_local = V_try.copy()
                        best_W_local = W_try.copy()
                else:
                    # Revert
                    mat[r_idx, c_idx] = old_val
                    
                # Occasionally, try a "jump" to escape local minima
                if step % 50 == 0:
                    # Perturb multiple variables
                    for _ in range(5):
                        mc = np.random.choice(3)
                        if mc == 0:
                            mat = U_try
                            dims = (n*m, target_rank)
                        elif mc == 1:
                            mat = V_try
                            dims = (m*p, target_rank)
                        else:
                            mat = W_try
                            dims = (p*n, target_rank)
                            
                        ri = np.random.randint(0, dims[0])
                        ci = np.random.randint(0, dims[1])
                        mat[ri, ci] = np.random.choice(discrete_vals)
                        
            if found:
                best_U = U_try.copy()
                best_V = V_try.copy()
                best_W = W_try.copy()
                best_rank = target_rank
                break

    # Convert to list of lists with rounding
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Clean up near-zero and near-integer/half-integer values
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
                elif abs(v - round(v*2)/2) < 1e-10:
                    row[idx] = round(v*2)/2
        return lst

    U_out = clean_list(to_list(best_U))
    V_out = clean_list(to_list(best_V))
    W_out = clean_list(to_list(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
