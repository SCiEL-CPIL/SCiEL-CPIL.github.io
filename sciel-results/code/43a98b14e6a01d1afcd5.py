import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for 120s limit
    
    # Schoolbook rank
    R_schoolbook = n * m * p
    
    # Lower bound
    R_min = max(n * m, m * p, p * n)
    
    best_solution = None
    best_rank = R_schoolbook
    
    # We will search from R_min up to R_schoolbook
    # But we prioritize finding a low rank.
    # If we don't find one quickly, we return schoolbook.
    
    # Pre-allocate or use efficient loops
    
    for R in range(R_min, R_schoolbook + 1):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Use a fixed seed for reproducibility per rank
        np.random.seed(42 + R)
        
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Reshape for efficient einsum
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # ALS Iterations
        # Limit iterations to keep time low
        max_iters = 50 if R < 50 else 20
        
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                break
                
            # Update U
            # A_u[i, j, r] = sum_k V[j, k, r] * W[k, i, r]
            # Using einsum: 'jkr,kir->ijr'
            A_u = np.einsum('jkr,kir->ijr', V_reshaped, W_reshaped)
            
            # U[i, j, r] should satisfy sum_r U[i, j, r] * A_u[i, j, r] = 1
            # Optimal U[i, j, :] = A_u[i, j, :] / (A_u[i, j, :] . A_u[i, j, :])
            norms_sq_u = np.sum(A_u * A_u, axis=2, keepdims=True)
            norms_sq_u[norms_sq_u == 0] = 1e-9
            U_new = A_u / norms_sq_u
            U_reshaped = U_new
            U = U_new.reshape(n * m, R)
            
            # Update V
            # A_v[j, k, r] = sum_i U[i, j, r] * W[k, i, r]
            # Using einsum: 'ijr,kir->jkr'
            A_v = np.einsum('ijr,kir->jkr', U_reshaped, W_reshaped)
            
            norms_sq_v = np.sum(A_v * A_v, axis=2, keepdims=True)
            norms_sq_v[norms_sq_v == 0] = 1e-9
            V_new = A_v / norms_sq_v
            V_reshaped = V_new
            V = V_new.reshape(m * p, R)
            
            # Update W
            # A_w[k, i, r] = sum_j U[i, j, r] * V[j, k, r]
            # Using einsum: 'ijr,jkr->kir'
            A_w = np.einsum('ijr,jkr->kir', U_reshaped, V_reshaped)
            
            norms_sq_w = np.sum(A_w * A_w, axis=2, keepdims=True)
            norms_sq_w[norms_sq_w == 0] = 1e-9
            W_new = A_w / norms_sq_w
            W_reshaped = W_new
            W = W_new.reshape(p * n, R)
            
            # Calculate Error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            # Term = einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            term = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            err = np.sum((term - 1.0) ** 2)
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                
            if best_error < 1e-8:
                break
                
        if best_error < 1e-6:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            # Reshape for verification
            U_int_r = U_int.reshape(n, m, R)
            V_int_r = V_int.reshape(m, p, R)
            W_int_r = W_int.reshape(p, n, R)
            
            term = np.einsum('ijr,jkr,kir->ijk', U_int_r, V_int_r, W_int_r)
            is_exact = np.all(np.abs(term - 1.0) < 1e-9)
            
            if is_exact:
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }

    # Fallback to schoolbook
    rank = R_schoolbook
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
