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

def _verify(U, V, W, n, m, p, tol=1e-6):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    Target = _build_target_tensor(n, m, p)
    err = np.max(np.abs(T_recon - Target))
    return err < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
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
    
    # Try ranks from 41 down to 35
    # The known lower bound is higher, but let's try to find something better than 45.
    # Actually, for 3x3x5, the rank is likely 41 or 42.
    
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        
        # Try multiple random restarts for this rank
        # We use more restarts for higher ranks initially, then reduce if time is tight
        max_restarts = 20 if target_rank > 38 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Use different seeds for diversity
            np.random.seed(123 + restart * 7 + target_rank)
            
            # Initialize with random values
            # Scale to roughly match the magnitude
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Adaptive gradient descent with restarts
            lr = 0.01
            best_err = np.inf
            stuck_count = 0
            
            for step in range(3000):
                if time.time() > deadline - 5:
                    break
                    
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T_target
                err = np.linalg.norm(diff)
                
                if err < best_err:
                    best_err = err
                    stuck_count = 0
                else:
                    stuck_count += 1
                    
                if stuck_count > 200:
                    # Restart with new random init or break
                    break
                
                # Gradient w.r.t U
                G_U = np.einsum('abc,br,cr->ar', diff, V, W)
                G_V = np.einsum('abc,ar,cr->br', diff, U, W)
                G_W = np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Adaptive learning rate
                grad_norm = np.linalg.norm(G_U) + np.linalg.norm(G_V) + np.linalg.norm(G_W)
                if grad_norm > 0:
                    lr_eff = lr * (1.0 + 0.1 * np.exp(-step/100))
                else:
                    lr_eff = lr
                    
                U -= lr_eff * G_U
                V -= lr_eff * G_V
                W -= lr_eff * G_W
                
                if step % 100 == 0 and err < 1e-4:
                    break
                        
            # Check if converged
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_recon - T_target)
            
            if err < 1e-4:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if _verify(U_round, V_round, W_round, n, m, p):
                    best_U, best_V, best_W = U_round, V_round, W_round
                    best_rank = target_rank
                    found = True
                    break # Found a valid one for this rank, try lower
        
        if found:
            # We found a decomposition for this rank, try to go lower
            continue
        else:
            # If we didn't find one for this rank, proceed to next lower rank
            pass

    # Convert to lists for output
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
