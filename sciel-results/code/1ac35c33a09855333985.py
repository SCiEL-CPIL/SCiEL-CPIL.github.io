import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Build the target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    
    start_time = time.time()
    timeout = 140  # Safety margin
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def als_decompose(target_rank, max_iters=300, tol=1e-9, seed=None):
        """Alternating Least Squares for tensor decomposition with better initialization."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Better initialization: use structured patterns
        # Initialize with small random values plus structured components
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Add structured initialization based on schoolbook pattern
        for r in range(target_rank):
            # Try different structured initializations
            if r % 3 == 0:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i + j + k) % target_rank == r:
                                U[i * m + j, r] += 1.0
                                V[j * p + k, r] += 1.0
                                W[k * n + i, r] += 1.0
            elif r % 3 == 1:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i * 3 + j * 2 + k) % target_rank == r:
                                U[i * m + j, r] += 1.0
                                V[j * p + k, r] += 1.0
                                W[k * n + i, r] += 1.0
            else:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i * 2 + j * 3 + k * 2) % target_rank == r:
                                U[i * m + j, r] += 1.0
                                V[j * p + k, r] += 1.0
                                W[k * n + i, r] += 1.0
        
        for iter_ in range(max_iters):
            if time.time() - start_time > timeout:
                return None, None, None, None
                
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    A = np.zeros((target_rank, p))
                    for k in range(p):
                        A[:, k] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    A = np.zeros((target_rank, n))
                    for i in range(n):
                        A[:, i] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    A = np.zeros((target_rank, m))
                    for j in range(m):
                        A[:, j] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Check convergence
            err = compute_error(U, V, W, target_rank)
            if err < tol:
                return U, V, W, err
                
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err
    
    # Try ranks from 14 upward
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            U, V, W, err = als_decompose(target_rank, max_iters=200, tol=1e-8, seed=restart)
            
            if U is None:
                break
                
            if err < 1e-6:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
                
        if found:
            continue
        else:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
