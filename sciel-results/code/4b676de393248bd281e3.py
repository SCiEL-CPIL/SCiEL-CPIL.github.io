import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 110s for safety
    
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    max_rank = schoolbook_rank
    
    # Precompute index arrays for speed
    # For U: indices for each (i,j) - we need V[j*p+k] and W[k*n+i] for k in range(p)
    U_V_idx = []  # For each (i,j), list of V row indices
    U_W_idx = []  # For each (i,j), list of W row indices
    for i in range(n):
        for j in range(m):
            U_V_idx.append(list(range(j * p, (j + 1) * p)))
            U_W_idx.append([k * n + i for k in range(p)])
    
    # For V: indices for each (j,k) - we need U[i*m+j] and W[k*n+i] for i in range(n)
    V_U_idx = []
    V_W_idx = []
    for j in range(m):
        for k in range(p):
            V_U_idx.append([i * m + j for i in range(n)])
            V_W_idx.append([k * n + i for i in range(n)])
    
    # For W: indices for each (k,i) - we need U[i*m+j] and V[j*p+k] for j in range(m)
    W_U_idx = []
    W_V_idx = []
    for k in range(p):
        for i in range(n):
            W_U_idx.append([i * m + j for j in range(m)])
            W_V_idx.append([j * p + k for j in range(m)])
    
    # Convert to arrays for faster indexing
    U_V_idx_arr = np.array(U_V_idx)
    U_W_idx_arr = np.array(U_W_idx)
    V_U_idx_arr = np.array(V_U_idx)
    V_W_idx_arr = np.array(V_W_idx)
    W_U_idx_arr = np.array(W_U_idx)
    W_V_idx_arr = np.array(W_V_idx)
    
    for R in range(min_rank, max_rank + 1):
        # Check deadline
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random initializations
        found_exact = False
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init_seed in range(42, 42 + 5):  # 5 random initializations
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(init_seed + R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(50):  # Reduced iterations, check deadline
                if time.time() > deadline - 10.0:
                    break
                
                # Solve for U: A_u[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r]
                A_u = np.zeros((n * m, R))
                for ij in range(n * m):
                    v_idx = U_V_idx_arr[ij]
                    w_idx = U_W_idx_arr[ij]
                    A_u[ij, :] = np.sum(V[v_idx, :] * W[w_idx, :], axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-9
                U = A_u / norms_sq[:, np.newaxis]
                
                # Solve for V: A_v[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
                A_v = np.zeros((m * p, R))
                for jk in range(m * p):
                    u_idx = V_U_idx_arr[jk]
                    w_idx = V_W_idx_arr[jk]
                    A_v[jk, :] = np.sum(U[u_idx, :] * W[w_idx, :], axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-9
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Solve for W: A_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                A_w = np.zeros((p * n, R))
                for ki in range(p * n):
                    u_idx = W_U_idx_arr[ki]
                    v_idx = W_V_idx_arr[ki]
                    A_w[ki, :] = np.sum(U[u_idx, :] * V[v_idx, :], axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w == 0] = 1e-9
                W = A_w / norms_sq_w[:, np.newaxis]
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_error < 1e-8:
                    break
            
            # Check if we found an exact solution
            if best_error < 1e-6:
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # If not exact at this rank, try next rank

    # Fallback to schoolbook if no low-rank solution found
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
