import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Lower bound: max(n*m, m*p, p*n) = 16
    # We focus on ranks close to schoolbook first, as they are more likely to exist
    # and be found by local methods. Ranks 40-47.
    ranks_to_try = list(range(47, 39, -1)) # 47 down to 40
    
    # Precompute tensor reshapes for efficient ALS
    T_UM = T.reshape(n * m, m * p * p * n)
    T_VM = T.reshape(m * p, n * m * p * n)  
    T_WM = T.reshape(p * n, n * m * m * p)
    
    max_trials_per_rank = 50
    max_als_iter = 30
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(max_trials_per_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(12345 + R * 1000 + trial)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize to prevent overflow/underflow in early iterations
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
            
            prev_error = np.inf
            converged = False
            
            for it in range(max_als_iter):
                if time.time() > deadline - 3:
                    break
                
                # Optimize U
                VW = np.einsum('jr,kr->rjk', V, W).reshape(R, m * p * p * n)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW.T, T_UM, rcond=None)
                    U = U_new
                except:
                    pass
                
                # Optimize V
                UW = np.einsum('ir,kr->rik', U, W).reshape(R, n * m * p * n)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW.T, T_VM, rcond=None)
                    V = V_new
                except:
                    pass
                
                # Optimize W
                UV = np.einsum('ir,jr->rij', U, V).reshape(R, n * m * m * p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV.T, T_WM, rcond=None)
                    W = W_new
                except:
                    pass
                
                # Compute error every few iterations
                if it % 5 == 0 or it == max_als_iter - 1:
                    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                    error = np.linalg.norm(T - recon)
                    if error < prev_error:
                        prev_error = error
                    if error < 1e-10:
                        converged = True
                        break
                    if abs(prev_error - error) < 1e-12:
                        break
            
            # Check for exact half-integer solution
            # First, check if the floating point solution is already close enough
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T, recon, atol=1e-8):
                # Try to snap to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                recon_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T, recon_rounded, atol=1e-9):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
