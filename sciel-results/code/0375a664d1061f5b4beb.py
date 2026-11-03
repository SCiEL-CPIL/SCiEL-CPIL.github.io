import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline to be safe within 120s limit
    
    # Precompute indices for U, V, W updates
    # For U[I, :]: I = i*m + j
    # System: sum_r U[I, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k in 0..p-1
    # Matrix B_u[I] is p x R. B_u[I, k, r] = V[j*p+k, r] * W[k*n+i, r]
    
    # For V[J, :]: J = j*p + k
    # System: sum_r V[J, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for i in 0..n-1
    # Matrix B_v[J] is n x R. B_v[J, i, r] = U[i*m+j, r] * W[k*n+i, r]
    
    # For W[K, :]: K = k*n + i
    # System: sum_r W[K, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for j in 0..m-1
    # Matrix B_w[K] is m x R. B_w[K, j, r] = U[i*m+j, r] * V[j*p+k, r]

    # Precompute index lists for faster access
    # For each I, store (j, list of W indices)
    u_indices = []
    for i in range(n):
        for j in range(m):
            I = i * m + j
            v_rows = np.arange(j * p, (j + 1) * p)
            w_rows = np.array([k * n + i for k in range(p)])
            u_indices.append((v_rows, w_rows))
            
    # For each J, store (j, k, list of U indices, list of W indices)
    v_indices = []
    for j in range(m):
        for k in range(p):
            J = j * p + k
            u_rows = np.array([i * m + j for i in range(n)])
            w_rows = np.array([k * n + i for i in range(n)])
            v_indices.append((u_rows, w_rows))
            
    # For each K, store (k, i, list of U indices, list of V indices)
    w_indices = []
    for k in range(p):
        for i in range(n):
            K = k * n + i
            u_rows = np.array([i * m + j for j in range(m)])
            v_rows = np.array([j * p + k for j in range(m)])
            w_indices.append((u_rows, v_rows))

    # Search ranks from a reasonable lower bound
    # Lower bound is max(nm, mp, pn) = 25. 
    # Known rank for 4x5x5 is likely < 100.
    # Try ranks 30 to 60.
    best_solution = None
    best_rank = n * m * p
    
    # Fallback solution (Schoolbook)
    U_fb = np.zeros((n * m, best_rank))
    V_fb = np.zeros((m * p, best_rank))
    W_fb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    # Try ranks
    for R in range(30, 61):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        max_restarts = 10
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 100 + restart)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            # ALS Iterations
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each I, solve B_u[I] @ U[I, :]^T = 1
                for I in range(n * m):
                    v_rows, w_rows = u_indices[I]
                    # B is p x R
                    # B[k, r] = V[v_rows[k], r] * W[w_rows[k], r]
                    B = V[v_rows, :] * W[w_rows, :]
                    # Solve B x = 1
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    U[I, :] = x
                    
                # Update V
                for J in range(m * p):
                    u_rows, w_rows = v_indices[J]
                    B = U[u_rows, :] * W[w_rows, :]
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[J, :] = x
                    
                # Update W
                for K in range(p * n):
                    u_rows, v_rows = w_indices[K]
                    B = U[u_rows, :] * V[v_rows, :]
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W[K, :] = x
                
                # Calculate Error
                # err = sum_{i,j,k} (sum_r U[I,r]V[J,r]W[K,r] - 1)^2
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        I = i * m + j
                        for k in range(p):
                            J = j * p + k
                            K = k * n + i
                            val = np.dot(U[I, :], V[J, :] * W[K, :])
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_error < 1e-8:
                    break
            
            if best_error < 1e-6:
                # Round to half-integers
                U_int = np.round(best_U * 2) / 2
                V_int = np.round(best_V * 2) / 2
                W_int = np.round(best_W * 2) / 2
                
                # Verify Exactness
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        I = i * m + j
                        for k in range(p):
                            J = j * p + k
                            K = k * n + i
                            val = np.dot(U_int[I, :], V_int[J, :] * W_int[K, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact: break
                    if not is_exact: break
                    
                if is_exact:
                    # Found a valid solution with rank R
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        # If no solution found for this R, continue to next R
        # We keep the best solution found so far in case we need to return something non-schoolbook
        # But the task asks for rank-minimal. If we don't find one < 100, we return schoolbook.
        
    # Fallback
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
# EVOLVE-BLOCK-END
