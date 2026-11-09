import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 84
    
    # Schoolbook solution
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
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W, tol=1e-6):
        """Check exact decomposition."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def tensor_error(U, V, W):
        """Compute sum of squared errors."""
        n, m, p = DIMENSIONS
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def als_step(U, V, W, n, m, p, r):
        """One ALS iteration: update U, V, W in sequence."""
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)
        
        # Update U: for each row i*m+j, solve least squares
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U_row, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = U_row
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # Update V: for each row j*p+k
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
                try:
                    V_row, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = V_row
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        # Update W: for each row k*n+i
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
                try:
                    W_row, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = W_row
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        
        return U_new, V_new, W_new
    
    # Strategy: Try to find lower-rank decompositions
    # Use multiple random initializations and ALS
    
    # First, try to find a solution with rank < 84
    # Known lower bounds: For 3x4x7, the minimum rank is at least 3*4 + 4*7 - 1 = 39 (from Strassen-like bounds)
    # Actually, for matrix multiplication A(m x n) * B(n x p), lower bound is max(m*n, m*p, n*p) but that's not tight
    # Better lower bound: rank >= max(n*m, m*p, n*p) is not correct
    # The actual lower bound for 3x4x7 is not well known, but it's definitely less than 84
    
    # Try various target ranks with multiple restarts
    target_ranks = list(range(83, 70, -1))  # Try from 83 down to 71
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        n_trials = 12
        
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            # Different random seeds for diversity
            seed = 42 + trial * 7919 + target_rank * 13
            np.random.seed(seed)
            
            # Initialize with random Gaussian
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            # Also try initialization from truncated schoolbook
            if trial % 3 == 1:
                U_init = U_school[:, :target_rank].copy()
                V_init = V_school[:, :target_rank].copy()
                W_init = W_school[:, :target_rank].copy()
                U_init += np.random.randn(n*m, target_rank) * 0.05
                V_init += np.random.randn(m*p, target_rank) * 0.05
                W_init += np.random.randn(p*n, target_rank) * 0.05
            elif trial % 3 == 2:
                # Mix: half schoolbook, half random
                half = target_rank // 2
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                U_init[:, :half] = U_school[:, :half]
                V_init[:, :half] = V_school[:, :half]
                W_init[:, :half] = W_school[:, :half]
                U_init[:, half:] = np.random.randn(n*m, target_rank - half) * 0.5
                V_init[:, half:] = np.random.randn(m*p, target_rank - half) * 0.5
                W_init[:, half:] = np.random.randn(p*n, target_rank - half) * 0.5
            
            # Run ALS with early stopping
            U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = tensor_error(U_cur, V_cur, W_cur)
            
            max_iters = 300
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                U_cur, V_cur, W_cur = als_step(U_cur, V_cur, W_cur, n, m, p, target_rank)
                curr_err = tensor_error(U_cur, V_cur, W_cur)
                if abs(prev_err - curr_err) < 1e-15:
                    break
                prev_err = curr_err
            
            # Check if we found a valid solution
            if verify(U_cur, V_cur, W_cur, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we can't find a solution at this rank, we might be able to go lower
            # but let's try a few more ranks
            if target_rank - 83 < -5:
                break
    
    # Final polish if we found an improvement
    if best_rank < rank_school:
        U_pol, V_pol, W_pol = best_U.copy(), best_V.copy(), best_W.copy()
        for it in range(200):
            if time.time() > deadline - 2:
                break
            U_pol, V_pol, W_pol = als_step(U_pol, V_pol, W_pol, n, m, p, best_rank)
        
        if verify(U_pol, V_pol, W_pol, tol=1e-8):
            best_U, best_V, best_W = U_pol, V_pol, W_pol
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
