import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    target_tensor = np.zeros((n*m, m*p, p*n))
    # Construct the matrix multiplication tensor
    # T[(i,j), (j,k), (k,i)] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i*m + j, j*p + k, k*n + i] = 1.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Store schoolbook as fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit
    
    # Try to find lower rank
    # We limit the search to a reasonable number of ranks to ensure completion
    max_ranks_to_try = 30
    
    for rank in range(best_rank - 1, max(best_rank - 1 - max_ranks_to_try, 0), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts for this rank
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
                
            # Initialization: Warm start from schoolbook for better convergence
            if rank < best_rank:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                # Copy first rank columns from schoolbook
                U[:, :rank] = U_school[:, :rank]
                V[:, :rank] = V_school[:, :rank]
                W[:, :rank] = W_school[:, :rank]
                
                # Add small noise to break symmetry and allow optimization
                U += np.random.randn(n*m, rank) * 1e-3
                V += np.random.randn(m*p, rank) * 1e-3
                W += np.random.randn(p*n, rank) * 1e-3
            else:
                U = np.random.randn(n*m, rank)
                V = np.random.randn(m*p, rank)
                W = np.random.randn(p*n, rank)
            
            # Vectorized ALS iterations
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # M1[a,k,c] = sum_b T[a,b,k] V[b,c]
                M1 = np.einsum('abk,bc->akc', target_tensor, V)
                # U[a,c] = sum_k M1[a,k,c] W[k,c]
                U_new = np.einsum('akc,kc->ac', M1, W)
                
                # Update V
                # M2[b,k,c] = sum_a T[a,b,k] U[a,c]
                M2 = np.einsum('abk,ac->bkc', target_tensor, U)
                # V[b,c] = sum_k M2[b,k,c] W[k,c]
                V_new = np.einsum('bkc,kc->bc', M2, W)
                
                # Update W
                # W[k,c] = sum_a M1[a,k,c] U[a,c]
                W_new = np.einsum('akc,ac->kc', M1, U)
                
                U = U_new
                V = V_new
                W = W_new
            
            # Check residual
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(target_tensor - T_approx)
            
            if residual < 1e-6:
                # Try to round to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                residual_rounded = np.linalg.norm(target_tensor - T_approx_rounded)
                
                if residual_rounded < 1e-9:
                    best_rank = rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break # Try next rank
        
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
