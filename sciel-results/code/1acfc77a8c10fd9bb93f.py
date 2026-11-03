import time
_start_time = time.time()
_TIMEOUT = 145.0

def construct():
    n, m, p = 2, 4, 8
    import numpy as np
    
    # Build the 56-term Strassen decomposition
    total_rank = 56
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1
            
            a_map = {
                0: 0 * m + (2 * half + 0),
                1: 0 * m + (2 * half + 1),
                2: 1 * m + (2 * half + 0),
                3: 1 * m + (2 * half + 1)
            }
            
            b_map = {
                0: (2 * half + 0) * p + (2 * jj + 0),
                1: (2 * half + 0) * p + (2 * jj + 1),
                2: (2 * half + 1) * p + (2 * jj + 0),
                3: (2 * half + 1) * p + (2 * jj + 1)
            }
            
            c_map = {
                0: (2 * jj + 0) * n + 0,
                1: (2 * jj + 1) * n + 0,
                2: (2 * jj + 0) * n + 1,
                3: (2 * jj + 1) * n + 1
            }

            for r in range(7):
                rr = base_r + r
                u_list, v_list, w_list = strassen_terms[r]
                
                for idx, val in u_list:
                    U[a_map[idx], rr] += val
                for idx, val in v_list:
                    V[b_map[idx], rr] += val
                for idx, val in w_list:
                    W[c_map[idx], rr] += val

    # Compute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for k in range(total_rank):
        T_target += np.einsum('i,j,l->ijl', U[:, k], V[:, k], W[:, k])

    # Try ALS to find lower-rank decomposition
    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for target_r in range(50, 43, -1):
        if time.time() - _start_time > _TIMEOUT - 15:
            break
        
        found = False
        for trial in range(3):
            if time.time() - _start_time > _TIMEOUT - 15:
                break
            
            np.random.seed(123 + trial)
            U_try = np.random.randn(n*m, target_r) * 0.1
            V_try = np.random.randn(m*p, target_r) * 0.1
            W_try = np.random.randn(p*n, target_r) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() - _start_time > _TIMEOUT - 15:
                    break
                
                # Fix V, W, solve for U
                for i in range(n*m):
                    # T_target[i,:,:] = sum_k U[i,k] * V[:,k] * W[:,k] (outer product)
                    # T_target[i,j,l] = sum_k U[i,k] * V[j,k] * W[l,k]
                    # For fixed i: T_target[i,:,:] = sum_k U[i,k] * (V[:,k] ⊗ W[:,k])
                    # This is a linear system in U[i,:]
                    # T_target[i,j,l] = sum_k U[i,k] * V[j,k] * W[l,k]
                    # Reshape: for each k, the contribution is U[i,k] * (V[:,k] * W[:,k]) (element-wise)
                    # So T_target[i,:,:] = sum_k U[i,k] * (V[:,k] ⊙ W[:,k]).reshape(m*p, p*n)
                    # where ⊙ is element-wise product, and we need to reshape
                    # V[:,k] is (m*p,), W[:,k] is (p*n,)
                    # V[:,k] ⊗ W[:,k] is (m*p, p*n)
                    # So T_target[i,:,:] = sum_k U[i,k] * (V[:,k] outer W[:,k])
                    # This is: T_target[i].reshape(-1) = sum_k U[i,k] * (V[:,k] * W[:,k]).reshape(-1)
                    # where V[:,k] * W[:,k] means outer product flattened
                    
                    # Build the design matrix
                    # X[i] is (m*p*p*n, target_r)
                    # X[i, j*p*n + l, k] = V[j,k] * W[l,k]
                    # T_target[i].reshape(-1) = X[i] @ U[i,:]
                    
                    X = np.zeros((m*p*p*n, target_r))
                    for k in range(target_r):
                        X[:, k] = (V_try[:, k] * W_try[:, k]).reshape(-1)
                    # Wait, that's not right. V[:,k] has length m*p, W[:,k] has length p*n
                    # Their outer product has shape (m*p, p*n), flattened to (m*p*p*n,)
                    # X[:, k] = np.outer(V_try[:, k], W_try[:, k]).flatten()
                    
                    X = np.zeros((m*p*p*n, target_r))
                    for k in range(target_r):
                        X[:, k] = np.outer(V_try[:, k], W_try[:, k]).flatten()
                    
                    y = T_target[i].flatten()
                    # Solve X @ U[i,:] = y
                    U_try[i, :], residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                
                # Fix U, W, solve for V
                for j in range(m*p):
                    X = np.zeros((n*m*p*n, target_r))
                    for k in range(target_r):
                        X[:, k] = np.outer(U_try[:, k], W_try[:, k]).flatten()
                    y = T_target[:, j, :].flatten()
                    V_try[j, :], residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                
                # Fix U, V, solve for W
                for l in range(p*n):
                    X = np.zeros((n*m*m*p, target_r))
                    for k in range(target_r):
                        X[:, k] = np.outer(U_try[:, k], V_try[:, k]).flatten()
                    y = T_target[:, :, l].flatten()
                    W_try[l, :], residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                
                # Check convergence
                T_try = np.zeros((n*m, m*p, p*n))
                for k in range(target_r):
                    T_try += np.einsum('i,j,l->ijl', U_try[:, k], V_try[:, k], W_try[:, k])
                err = np.sum((T_target - T_try) ** 2)
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                best_rank = target_r
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
