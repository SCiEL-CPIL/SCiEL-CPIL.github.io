"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N, M, P = n, m, p  # aliases

# EVOLVE-BLOCK-START

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, tol=1e-6):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape for einsum
    # U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    
    U_r = U.reshape((n, m, rank))
    V_r = V.reshape((m, p, rank))
    W_r = W.reshape((p, n, rank))
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    actual = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    
    diff = np.max(np.abs(actual - target))
    return diff < tol, diff

def _try_reduce_rank(schoolbook_result, max_rank_target, time_budget=30.0):
    """Try to find a lower-rank decomposition by zeroing out columns and checking."""
    U = [row[:] for row in schoolbook_result["U"]]
    V = [row[:] for row in schoolbook_result["V"]]
    W = [row[:] for row in schoolbook_result["W"]]
    rank = len(U[0])
    
    start_time = time.time()
    best_rank = rank
    best_result = None
    
    # Greedy: try to remove columns one by one
    # For each column, check if it's redundant
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break
        if best_rank <= max_rank_target:
            break
        
        # Shuffle column order for diversity
        cols = list(range(best_rank))
        random.shuffle(cols)
        
        for c in cols:
            if time.time() - start_time > time_budget:
                break
            # Try removing column c
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            
            valid, _ = _check_decomposition(U_test, V_test, W_test)
            if valid:
                U = U_test
                V = V_test
                W = W_test
                best_rank -= 1
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result if best_result else schoolbook_result

def _optimize_search(target_rank, time_budget=30.0):
    """Optimization-based search for a decomposition of a specific rank."""
    start_time = time.time()
    
    # Try a few random seeds
    for seed in range(50):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(seed)
        
        # Initialize with small integers
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        W = np.random.randint(-1, 2, size=(p*n, target_rank))
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank :]
            
            U = U_flat.reshape((n, m, target_rank))
            V = V_flat.reshape((m, p, target_rank))
            W = W_flat.reshape((p, n, target_rank))
            
            # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            error = T - 1.0
            return np.sum(error**2)
        
        try:
            # Use Nelder-Mead
            res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 1e-4:
                x_opt = res.x
                U_opt = x_opt[:n*m*target_rank].reshape((n*m, target_rank))
                V_opt = x_opt[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape((m*p, target_rank))
                W_opt = x_opt[n*m*target_rank + m*p*target_rank :].reshape((p*n, target_rank))
                
                # Try rounding to integers
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                
                valid, diff = _check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist())
                if valid:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rounding to half-integers
                U_half = np.round(2 * U_opt) / 2.0
                V_half = np.round(2 * V_opt) / 2.0
                W_half = np.round(2 * W_opt) / 2.0
                
                valid, diff = _check_decomposition(U_half.tolist(), V_half.tolist(), W_half.tolist())
                if valid:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                    
        except Exception:
            pass
            
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Try optimization search for lower ranks
    # Known lower bound is around 12 for 2x3x3
    for rank in range(12, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on remaining budget
        time_left = deadline - time.time() - 5
        if time_left < 5:
            break
            
        budget = min(time_left / (best_rank - rank), 20.0) # Spread time
        
        opt_res = _optimize_search(rank, time_budget=budget)
        if opt_res:
            best_rank = rank
            best_result = opt_res
            break # Found a valid one, stop searching for lower ranks immediately? 
            # Or continue to see if we can go lower? 
            # Since we are iterating from 12 upwards, if we find 12, it's likely minimal or close.
            # But let's break to save time for verification/return.
            
    return best_result

# EVOLVE-BLOCK-END
