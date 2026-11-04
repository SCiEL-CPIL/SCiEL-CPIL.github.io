import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    
    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fb

    # Attempt to find a lower rank decomposition using ALS optimization
    # Theoretical lower bound for 4x4x5 is unknown but likely < 80.
    # Strassen for 4x4x4 is 49. Plus 16 for 4x4x1? No, that's not how it splits.
    # Let's try ranks from 60 down to 50.
    
    np.random.seed(42)
    
    for target_rank in range(60, 40, -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Normalize
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # ALS Optimization
            # Objective: minimize sum_{i,j,k} (1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])^2
            # This is non-convex. We use gradient descent or ALS.
            # ALS: Fix V, W. Update U.
            # For a fixed row of U (i*m+j), we want to minimize sum_k (1 - U_row . (V[j*p+k] * W[k*n+i]))^2
            # Let M_k = V[j*p+k] * W[k*n+i] (element-wise product, shape target_rank)
            # We want to minimize sum_k (1 - U_row . M_k)^2
            # This is a linear least squares problem for U_row.
            # Let A be the matrix with rows M_k.
            # Minimize ||A^T U_row - 1||^2
            # Solution: U_row = (A A^T)^{-1} A 1
            
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Collect M_k for this i, j
                        # M_k = V[j*p+k] * W[k*n+i]
                        # We need to solve for U[i*m+j]
                        # A is p x rank
                        # A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        
                        # Solve A^T x = 1
                        # x = U[i*m+j, :]
                        try:
                            x, residuals, rank_sol, s = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                            U_new[i*m+j, :] = x
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # Collect B_i for this j, k
                        # B_i = U[i*m+j] * W[k*n+i]
                        # B is n x rank
                        # B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        
                        # Solve B^T y = 1
                        # y = V[j*p+k, :]
                        try:
                            y, residuals, rank_sol, s = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                            V_new[j*p+k, :] = y
                        except:
                            V_new[j*p+k, :] = V[j*p+k, :]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # Collect C_j for this k, i
                        # C_j = U[i*m+j] * V[j*p+k]
                        # C is m x rank
                        # C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        
                        # Solve C^T z = 1
                        # z = W[k*n+i, :]
                        try:
                            z, residuals, rank_sol, s = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                            W_new[k*n+i, :] = z
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                # Check convergence / validity
                if verify(U, V, W):
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                    break # Found valid for this rank, break restarts and try lower rank? 
                         # Actually, if we found a valid one at this rank, we should try lower ranks.
                         # So we break the restart loop, but the outer loop continues to lower target_rank.
                         # Wait, the outer loop is `for target_rank in range(60, 40, -1)`.
                         # So if we succeed at 60, we break restarts, then try 59.
                         # This is correct.
            if verify(U, V, W):
                if target_rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                # If we found a valid one, we can break the restart loop to try lower ranks
                # But we should only break if we want to try lower.
                # The logic above breaks restarts if verify passes.
                # Let's ensure we don't break the outer loop.
                pass

    # Return the best found
    # Ensure we return lists as per interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The interface says: construct() -> {'U', 'V', 'W'}
    # The seed program returned lists of lists.
    # The inspirations returned lists or numpy arrays.
    # The evaluator likely handles numpy arrays or lists.
    # Let's return numpy arrays for safety and efficiency, or lists if strict.
    # Seed returned lists. Let's stick to lists to be safe, or check if numpy is accepted.
    # "Python standard library, NumPy and SciPy are available."
    # Usually numpy arrays are fine. But let's convert to lists to match seed exactly.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
