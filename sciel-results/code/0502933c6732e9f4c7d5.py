import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    min_rank = max(n * m, m * p, p * n)
    
    # We try ranks starting from the lower bound upwards
    # Given time constraints, we try a few small ranks
    max_ranks_to_try = 20 # Try up to min_rank + 19
    base_rank = min_rank
    
    # Pre-compute indices for speed
    U_indices = [[i * m + j for j in range(m)] for i in range(n)]
    V_indices = [[j * p + k for k in range(p)] for j in range(m)]
    W_indices = [[k * n + i for i in range(n)] for k in range(p)]
    
    # For A_u calculation: for each (i,j), need V[j*p:(j+1)*p, :] and W[k*n+i for k in range(p)]
    # For A_v calculation: for each (j,k), need U[i*m+j for i in range(n)] and W[k*n+i for i in range(n)]
    # For A_w calculation: for each (k,i), need U[i*m+j for j in range(m)] and V[j*p+k for j in range(m)]
    
    for rank_offset in range(max_ranks_to_try):
        R = base_rank + rank_offset
        
        # Try multiple random restarts
        found_solution = False
        num_restarts = 5
        for restart in range(num_restarts):
            np.random.seed(42 + rank_offset * 100 + restart)
            # Initialize with random 0s and 1s? Or randn?
            # Let's try randn scaled down
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for it in range(100): # 100 iterations
                # Solve for U
                A_u = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        W_idx = np.array([k * n + i for k in range(p)])
                        W_slice = W[W_idx, :]
                        V_slice = V[j * p:(j + 1) * p, :]
                        A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-9
                U = A_u / norms_sq[:, np.newaxis]
                
                # Solve for V
                A_v = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_idx = np.array([i * m + j for i in range(n)])
                        U_slice = U[U_idx, :]
                        W_idx = np.array([k * n + i for i in range(n)])
                        W_slice = W[W_idx, :]
                        A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-9
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Solve for W
                A_w = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_idx = np.array([i * m + j for j in range(m)])
                        U_slice = U[U_idx, :]
                        V_idx = np.array([j * p + k for j in range(m)])
                        V_slice = V[V_idx, :]
                        A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
                
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
                    
                if best_error < 1e-6:
                    break
            
            if best_error < 1e-4:
                # Try rounding to half-integers
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
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not found in restarts, try next rank

    # Fallback to schoolbook
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
