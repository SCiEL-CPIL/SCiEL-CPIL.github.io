import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def check_error(U, V, W, R):
        """Check if the decomposition is exact."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :R], V[j * p + k, :R] * W[k * n + i, :R])
                    error += (val - 1.0) ** 2
        return error
    
    def try_round(U, V, W, R):
        """Try to round to half-integers and check."""
        U_rounded = np.round(U[:, :R] * 2) / 2
        V_rounded = np.round(V[:, :R] * 2) / 2
        W_rounded = np.round(W[:, :R] * 2) / 2
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                    error += (val - 1.0) ** 2
                    
        if error < 1e-6:
            return U_rounded, V_rounded, W_rounded
        return None
    
    def als_solve(U, V, W, R, max_iters=100):
        """ALS iterations to refine the decomposition."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Solve for U given V, W
            A_U = np.zeros((n * p, n * m * R))
            b_U = np.ones(n * p)
            
            for i in range(n):
                for k in range(p):
                    row_idx = k * n + i
                    for j in range(m):
                        for r in range(R):
                            col_idx = (i * m + j) * R + r
                            A_U[row_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
                            
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(n * m, R)
            except:
                break
                
            # Solve for V given U, W
            A_V = np.zeros((m * p, m * p * R))
            b_V = np.ones(m * p)
            
            for j in range(m):
                for k in range(p):
                    row_idx = j * p + k
                    for i in range(n):
                        for r in range(R):
                            col_idx = (j * p + k) * R + r
                            A_V[row_idx, col_idx] = U[i * m + j, r] * W[k * n + i, r]
                            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(m * p, R)
            except:
                break
                
            # Solve for W given U, V
            A_W = np.zeros((p * n, p * n * R))
            b_W = np.ones(p * n)
            
            for k in range(p):
                for i in range(n):
                    row_idx = k * n + i
                    for j in range(m):
                        for r in range(R):
                            col_idx = (k * n + i) * R + r
                            A_W[row_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
                            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(p * n, R)
            except:
                break
                
        return U, V, W
    
    # Try ranks from schoolbook-1 down to a minimum
    # Focus on a few promising ranks with multiple restarts
    target_ranks = list(range(rank_schoolbook - 1, rank_schoolbook - 20, -1))
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with schoolbook subset + noise
            U = U_sb[:, :R].copy()
            V = V_sb[:, :R].copy()
            W = W_sb[:, :R].copy()
            
            # Add random perturbation
            scale = 0.5 + restart * 0.1
            U += np.random.normal(0, scale, U.shape)
            V += np.random.normal(0, scale, V.shape)
            W += np.random.normal(0, scale, W.shape)
            
            # Run ALS
            U, V, W = als_solve(U, V, W, R, max_iters=50)
            
            # Check error
            error = check_error(U, V, W, R)
            
            if error < 1e-6:
                # Try to round
                rounded = try_round(U, V, W, R)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_rank = R
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    break  # Move to next rank
                
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
