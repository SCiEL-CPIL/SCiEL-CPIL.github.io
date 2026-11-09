import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition
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
                
    T = np.ones((n, m, p))
    
    # Target rank - try to find better than schoolbook
    target_rank = 20
    
    # Function to compute error
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    # Direct discrete search on half-integer grid
    # We'll use a local search approach with structured initializations
    
    best_solution = None
    best_rank = rank_school
    
    rng = np.random.default_rng(42)
    
    # Try different rank values, starting from lower ranks
    for rank in range(10, target_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with structured random values on half-integer grid
            # Values in {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            err = compute_error(U, V, W)
            
            if err < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
            
            # Local search: try flipping individual entries
            improved = True
            while improved and time.time() < deadline:
                improved = False
                err_before = err
                
                # Try flipping U entries
                for i in range(n*m):
                    for r in range(rank):
                        old_val = U[i, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            U[i, r] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < err_before - 1e-12:
                                err_before = new_err
                                err = new_err
                                improved = True
                            else:
                                U[i, r] = old_val
                        U[i, r] = old_val  # Restore
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
                
                # Try flipping V entries
                for j in range(m*p):
                    for r in range(rank):
                        old_val = V[j, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            V[j, r] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < err_before - 1e-12:
                                err_before = new_err
                                err = new_err
                                improved = True
                            else:
                                V[j, r] = old_val
                        V[j, r] = old_val
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
                
                # Try flipping W entries
                for k in range(p*n):
                    for r in range(rank):
                        old_val = W[k, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            W[k, r] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < err_before - 1e-12:
                                err_before = new_err
                                err = new_err
                                improved = True
                            else:
                                W[k, r] = old_val
                        W[k, r] = old_val
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
                
                if err < 1e-9:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        # If we found a solution with this rank, return it
        if best_solution is not None and rank < best_rank:
            best_rank = rank
    
    # If no better solution found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
