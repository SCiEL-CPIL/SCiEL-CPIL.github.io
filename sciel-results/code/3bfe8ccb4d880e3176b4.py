"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction as F
import numpy as np
from itertools import product
import scipy.optimize

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as fallback
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def validate_strict(U, V, W, rank):
        """Check with exact arithmetic using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = F(1)
                    actual = F(0)
                    for r in range(rank):
                        actual += F(U[i * m + j][r]) * F(V[j * p + k][r]) * F(W[k * n + i][r])
                    if actual != expected:
                        return False
        return True

    def error_function(params, rank):
        """Calculate the error for a given set of parameters."""
        # params is a flat array of length (n*m + m*p + p*n) * rank
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank : n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank :]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    error += (actual - 1.0) ** 2
        return error

    def try_rank(target_rank, num_restarts=5):
        if time.time() > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            random.seed(42 + target_rank * 100 + restart)
            
            # Initialize with small random values
            # We want to find a solution where entries are half-integers.
            # Start with random integers in {-1, 0, 1} scaled down?
            # Or just random floats.
            U_init = np.random.randint(-1, 2, (n * m, target_rank)).astype(float)
            V_init = np.random.randint(-1, 2, (m * p, target_rank)).astype(float)
            W_init = np.random.randint(-1, 2, (p * n, target_rank)).astype(float)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy to minimize error
            # We use 'L-BFGS-B' for speed and robustness
            try:
                result = scipy.optimize.minimize(
                    error_function, 
                    params_init, 
                    args=(target_rank,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'ftol': 1e-6}
                )
            except Exception:
                continue
                
            if not result.success:
                # Even if not fully successful, check if error is low
                pass
                
            params_opt = result.x
            U_opt = params_opt[:n * m * target_rank].reshape(n * m, target_rank)
            V_opt = params_opt[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
            W_opt = params_opt[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
            
            # Check error
            err = error_function(params_opt, target_rank)
            if err > 1e-3:
                continue
                
            # Snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            # Convert to lists of floats for validation
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            # Strict validation
            if validate_strict(U_list, V_list, W_list, target_rank):
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # If direct snap fails, try a local search around the snapped values
            # to fix any small errors
            U_curr = [row[:] for row in U_list]
            V_curr = [row[:] for row in V_list]
            W_curr = [row[:] for row in W_list]
            
            # Calculate current error
            def get_error(U, V, W):
                e = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(target_rank))
                            e += (a - 1.0)**2
                return e
                
            curr_err = get_error(U_curr, V_curr, W_curr)
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                if curr_err < 1e-12:
                    if validate_strict(U_curr, V_curr, W_curr, target_rank):
                        return {"U": U_curr, "V": V_curr, "W": W_curr}
                    break
                
                # Randomly pick a matrix and an entry to perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U_curr
                elif mat_idx == 1:
                    mat = V_curr
                else:
                    mat = W_curr
                    
                row_idx = random.randint(0, len(mat) - 1)
                col_idx = random.randint(0, target_rank - 1)
                
                old_val = mat[row_idx][col_idx]
                # Try neighbors: +/- 0.5
                new_val = old_val + random.choice([-0.5, 0.5])
                
                # Apply change
                mat[row_idx][col_idx] = new_val
                new_err = get_error(U_curr, V_curr, W_curr)
                
                if new_err < curr_err:
                    curr_err = new_err
                else:
                    # Revert
                    mat[row_idx][col_idx] = old_val
                    
    # Try ranks from 30 to 44
    # We prioritize lower ranks
    for target_rank in range(30, rank_schoolbook):
        if time.time() > deadline:
            break
            
        result = try_rank(target_rank, num_restarts=3)
        if result:
            best_rank = target_rank
            best_result = result
            break # Found a valid decomposition, stop
            
    # If no better rank found, return schoolbook
    if best_rank == rank_schoolbook:
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        
    return best_result
# EVOLVE-BLOCK-END
