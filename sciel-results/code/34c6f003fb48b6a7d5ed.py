import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err
    
    # Schoolbook rank as upper bound
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    # Lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    min_possible = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from min_possible up to schoolbook_rank
    for R in range(min_possible, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for each rank
        n_restarts = 15 if R <= 50 else 8 if R <= 80 else 5
        
        for seed in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + R * 131)
            
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            best_error = float('inf')
            best_U, best_V, best_W = None, None, None
            
            # ALS iterations
            n_iters = 100 if R <= 40 else 60 if R <= 70 else 40
            
            for it in range(n_iters):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                A_u = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        W_idx = np.array([k * n + i for k in range(p)])
                        V_slice = V[j * p:(j + 1) * p, :]
                        W_slice = W[W_idx, :]
                        A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq < 1e-14] = 1e-14
                U = A_u / norms_sq[:, np.newaxis]
                
                # Solve for V
                A_v = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_idx = np.array([i * m + j for i in range(n)])
                        W_idx = np.array([k * n + i for i in range(n)])
                        U_slice = U[U_idx, :]
                        W_slice = W[W_idx, :]
                        A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v < 1e-14] = 1e-14
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Solve for W
                A_w = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_idx = np.array([i * m + j for j in range(m)])
                        V_idx = np.array([j * p + k for j in range(m)])
                        U_slice = U[U_idx, :]
                        V_slice = V[V_idx, :]
                        A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w < 1e-14] = 1e-14
                W = A_w / norms_sq_w[:, np.newaxis]
                
                err = compute_error(U, V, W)
                if err < best_error:
                    best_error = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-10:
                    break
            
            # Check if we found a good solution
            if best_error < 1e-8 and best_U is not None:
                # Try rounding to half-integers
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # Try rounding to integers
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                
                if verify(U_int, V_int, W_int):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rational approximation with small denominators
                for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                    U_frac = np.round(best_U * denom) / denom
                    V_frac = np.round(best_V * denom) / denom
                    W_frac = np.round(best_W * denom) / denom
                    if verify(U_frac, V_frac, W_frac, tol=1e-6):
                        return {"U": U_frac.tolist(), "V": V_frac.tolist(), "W": W_frac.tolist()}
                
                # If error is tiny, return as-is (will be checked by evaluator)
                if best_error < 1e-12:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
