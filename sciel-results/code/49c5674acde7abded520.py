import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                Target[u, v, w] = 1.0
    err = np.max(np.abs(T - Target))
    return err < 1e-6

def _snap_to_half_int(M):
    return np.round(M * 2) / 2

def _local_discrete_search(U, V, W, T_target, n, m, p, max_iters=5000, time_limit=None):
    """
    Perform a local search on the discrete half-integer grid.
    Start from U, V, W (which should be close to a solution).
    """
    U = _snap_to_half_int(U)
    V = _snap_to_half_int(V)
    W = _snap_to_half_int(W)
    
    current_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_target)
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit and time.time() > time_limit:
            break
            
        # Pick a random matrix and a random entry
        mat_idx = np.random.choice(3)
        if mat_idx == 0:
            row, col = np.random.choice(U.shape[0]), np.random.choice(U.shape[1])
            old_val = U[row, col]
            # Try changing by +/- 0.5 or +/- 1.0
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            U[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                U[row, col] = old_val
        elif mat_idx == 1:
            row, col = np.random.choice(V.shape[0]), np.random.choice(V.shape[1])
            old_val = V[row, col]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            V[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                V[row, col] = old_val
        else:
            row, col = np.random.choice(W.shape[0]), np.random.choice(W.shape[1])
            old_val = W[row, col]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            W[row, col] = new_val
            T_new = np.einsum('ar,br,cr->abc', U, V, W)
            new_err = np.linalg.norm(T_new - T_target)
            if new_err < current_err:
                current_err = new_err
            else:
                W[row, col] = old_val
                
        if current_err < 1e-6:
            break
            
    return U, V, W, current_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_school[u, col] = 1
                V_school[v, col] = 1
                W_school[w, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    T_target = _build_target_tensor(n, m, p)
    
    # Known result for 3x3 * 3x5 matrix multiplication
    # The rank is known to be 39 (or close to it)
    # We will try ranks from 44 down to 35
    for target_rank in range(44, 34, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(100 + restart * 7)
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient Descent Phase
            lr = 0.05
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T_target
                
                G_U = np.einsum('abc,br,cr->ar', diff, V, W)
                G_V = np.einsum('abc,ar,cr->br', diff, U, W)
                G_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                if step % 50 == 0:
                    err = np.linalg.norm(diff)
                    if err < 1e-3:
                        break
                        
            # Discrete Search Phase
            U, V, W, err = _local_discrete_search(U, V, W, T_target, n, m, p, max_iters=2000, time_limit=deadline - 5)
            
            if err < 1e-6:
                # Verify strictly
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
                    
        if found:
            continue # Try lower rank
        else:
            # If not found, we might still try lower ranks if time permits, 
            # but usually if we fail at 44, we might fail at 43.
            # However, the search is stochastic.
            pass

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
