import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Equation: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # Try ranks from low to high
    # Start with a reasonable guess
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    for target_rank in [32, 24, 20, 16, 12, 10]:
        if time.time() > deadline - 30:
            break
        # Try random initialization and iterative refinement
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute residuals
                residuals = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residuals[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                res_norm = np.abs(residuals - 1).max()
                if res_norm < 1e-6:
                    # Found a solution, try to clean up to half-integers
                    break
                
                # Gradient step: for each equation, adjust U, V, W
                # d/dU[i*m+j, t] of residual[i,j,k] = V[j*p+k, t] * W[k*n+i, t]
                # We want to reduce |residual - 1|
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = residuals[i,j,k] - 1
                            if abs(err) < 1e-8:
                                continue
                            for t in range(target_rank):
                                u_idx = i*m+j
                                v_idx = j*p+k
                                w_idx = k*n+i
                                # Gradient of err w.r.t. U[u_idx, t]
                                grad_U = V[v_idx, t] * W[w_idx, t]
                                grad_V = U[u_idx, t] * W[w_idx, t]
                                grad_W = U[u_idx, t] * V[v_idx, t]
                                U[u_idx, t] -= lr * err * grad_U
                                V[v_idx, t] -= lr * err * grad_V
                                W[w_idx, t] -= lr * err * grad_W
                
                # Check convergence
                if it % 20 == 0:
                    residuals_check = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residuals_check[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if np.abs(residuals_check - 1).max() < 1e-6:
                        break
            
            # Final check
            residuals_final = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals_final[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            
            if np.abs(residuals_final - 1).max() < 1e-4:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Check if rounded version still works
                residuals_rounded = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residuals_rounded[i,j,k] = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                
                if np.abs(residuals_rounded - 1).max() < 1e-6:
                    U, V, W = U_rounded, V_rounded, W_rounded
                else:
                    # Use the unrounded but verify more carefully
                    pass
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break  # This rank works, try lower
    
    if time.time() > deadline - 5:
        pass  # Use best result
    
    return best_result
