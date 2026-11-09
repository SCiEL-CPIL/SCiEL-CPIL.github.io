# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Target tensor T
    T = np.ones((n, m, p))
    
    # Try ranks from lower bound up to best_rank - 1
    # Lower bound: let's use 32
    for target_rank in range(32, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try to find a decomposition of rank target_rank
        # Use multiple random restarts
        found = None
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            result = _greedy_cp_decomposition(T, target_rank, n, m, p, time_limit=2.0, seed=restart)
            if result is not None:
                U, V, W = result
                if _validate(U, V, W, n, m, p):
                    found = (U, V, W)
                    break
        
        if found is not None:
            best_rank = target_rank
            best_result = found
            # Since we are iterating from low to high, this is the best so far.
            # We can continue to see if we can find an even lower one? No, we started at 32.
            # If we found 50, we checked 32-49.
            # So we can break.
            break

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return (U, V, W)

def _validate(U, V, W, n, m, p):
    R = U.shape[1]
    T_recon = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape((n, m))
        v = V[:, r].reshape((m, p))
        w = W[:, r].reshape((p, n))
        for i in range(n):
            for j in range(m):
                u_val = u[i, j]
                if u_val == 0: continue
                for k in range(p):
                    T_recon[i, j, k] += u_val * v[j, k] * w[k, i]
                    
    T_target = np.ones((n, m, p))
    if not np.allclose(T_recon, T_target, atol=1e-6, rtol=1e-6):
        return False
        
    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    T_recon_rounded = np.zeros((n, m, p))
    for r in range(R):
        u = U_rounded[:, r].reshape((n, m))
        v = V_rounded[:, r].reshape((m, p))
        w = W_rounded[:, r].reshape((p, n))
        for i in range(n):
            for j in range(m):
                u_val = u[i, j]
                if u_val == 0: continue
                for k in range(p):
                    T_recon_rounded[i, j, k] += u_val * v[j, k] * w[k, i]
                    
    if np.allclose(T_recon_rounded, T_target, atol=1e-9, rtol=1e-9):
        return True
    return False

def _greedy_cp_decomposition(T, rank, n, m, p, time_limit=2.0, seed=0):
    start_time = time.time()
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    np.random.seed(seed)
    
    for r in range(rank):
        if time.time() > start_time + time_limit:
            return None
            
        u = np.random.rand(n, m)
        v = np.random.rand(m, p)
        w = np.random.rand(p, n)
        
        u /= (np.linalg.norm(u) + 1e-9)
        v /= (np.linalg.norm(v) + 1e-9)
        w /= (np.linalg.norm(w) + 1e-9)
        
        for it in range(200):
            u_new = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    vec = v[j, :] * w[:, i]
                    u_new[i, j] = np.dot(residual[i, j, :], vec)
            u_new /= (np.linalg.norm(u_new) + 1e-9)
            
            v_new = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    vec = u[:, j] * w[k, :]
                    v_new[j, k] = np.dot(residual[:, j, k], vec)
            v_new /= (np.linalg.norm(v_new) + 1e-9)
            
            w_new = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    vec = u[i, :] * v[:, k]
                    w_new[k, i] = np.dot(residual[i, :, k], vec)
            w_new /= (np.linalg.norm(w_new) + 1e-9)
            
            if np.allclose(u, u_new, atol=1e-8) and np.allclose(v, v_new, atol=1e-8) and np.allclose(w, w_new, atol=1e-8):
                break
                
            u, v, w = u_new, v_new, w_new
            
        term = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_val = u[i, j]
                if u_val == 0: continue
                for k in range(p):
                    term[i, j, k] = u_val * v[j, k] * w[k, i]
                    
        residual -= term
        U_list.append(u.flatten())
        V_list.append(v.flatten())
        W_list.append(w.flatten())
        
        if np.allclose(residual, 0, atol=1e-8):
            break
            
    if len(U_list) == 0:
        return None
        
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Check if the residual is small enough to be considered a solution
    # before validation
    if not np.allclose(residual, 0, atol=1e-6):
        return None
        
    return (U, V, W)
# EVOLVE-BLOCK-END
