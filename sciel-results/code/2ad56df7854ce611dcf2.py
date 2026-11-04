import numpy as np
import time
import itertools
import scipy.optimize

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution (Rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    def compute_error(U, V, W):
        """Compute Frobenius norm of T - sum_u U[:,u] \otimes V[:,u] \otimes W[:,u]"""
        A = np.einsum('iu,ju,ku->ijk', U, V, W)
        return np.linalg.norm(T - A)

    def try_optimize(rank, seed, max_iters=200):
        """Try to find a valid decomposition of given rank using scipy.optimize.least_squares."""
        np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Flatten parameters for optimizer
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residual(x):
            U_flat = x[:n*m*rank]
            V_flat = x[n*m*rank : n*m*rank + m*p*rank]
            W_flat = x[n*m*rank + m*p*rank :]
            
            U_mat = U_flat.reshape(n*m, rank)
            V_mat = V_flat.reshape(m*p, rank)
            W_mat = W_flat.reshape(p*n, rank)
            
            A = np.einsum('iu,ju,ku->ijk', U_mat, V_mat, W_mat)
            return (T - A).flatten()
        
        try:
            time_left = deadline - time.time()
            if time_left < 5:
                return None, float('inf')
            
            # Use least_squares which is robust for this
            res = scipy.optimize.least_squares(residual, x0, method='lm', max_nfev=2000)
            
            if res.cost < 1e-14: # Very small residual
                x_opt = res.x
                U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
                V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
                
                err = compute_error(U_opt, V_opt, W_opt)
                if err < 1e-7:
                    # Snap to half-integers
                    U_int = np.round(U_opt * 2) / 2
                    V_int = np.round(V_opt * 2) / 2
                    W_int = np.round(W_opt * 2) / 2
                    err_int = compute_error(U_int, V_int, W_int)
                    if err_int < 1e-9:
                        return (U_int, V_int, W_int), err_int
                    else:
                        # Try integers
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        err_int = compute_error(U_int, V_int, W_int)
                        if err_int < 1e-9:
                            return (U_int, V_int, W_int), err_int
                        else:
                            # Try to snap to nearest 0.5 grid more carefully?
                            # Or just return None
                            return None, err
                else:
                    return None, err
            else:
                return None, res.cost
                
        except Exception:
            return None, float('inf')

    # Search for lower ranks
    # Try 59, 58, 57
    # Allocate time wisely.
    
    for rank in range(rank_school - 1, max(rank_school - 5, 40), -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few seeds
        for seed in range(10):
            if time.time() > deadline - 10:
                break
                
            res, err = try_optimize(rank, seed, max_iters=200)
            if res is not None:
                best_rank = rank
                best_U, best_V, best_W = res
                found = True
                break # Found valid rank, try lower
                
        if found:
            continue
        else:
            # If we fail at 59, it's likely 60 is minimal.
            # Given the difficulty, break early if 59 fails.
            if rank == rank_school - 1:
                break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
