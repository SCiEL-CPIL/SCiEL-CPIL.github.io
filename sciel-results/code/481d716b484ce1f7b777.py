import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > tol:
                    return False
    return True

def try_rank(n, m, p, rank, max_iter=1000, tol=1e-9, seed=42):
    """Try to find a rank-r decomposition using iterative least squares."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update U: for each column i*m+j, solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                # Build linear system: sum_r U[u_idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # For fixed i,j, we have p equations
                # Let a_r = V[j*p+k, r] * W[k*n+i, r] for each k
                # We want: U[u_idx, :] dot a_r = 1 for each k
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                b_vec = np.ones(p)
                # Least squares
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[u_idx, :] = sol
        
        # Update V: for each column j*p+k, solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                b_vec = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[v_idx, :] = sol
        
        # Update W: for each column k*n+i, solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b_vec = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[w_idx, :] = sol
        
        # Check error
        error = 0.0
        for (u_idx, v_idx, w_idx) in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            error += (val - 1) ** 2
        error = np.sqrt(error / len(indices))
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            return best_U, best_V, best_W, error
    
    return best_U, best_V, best_W, best_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Seed: Rank 45
    rank_seed = n * m * p
    U = np.zeros((n * m, rank_seed))
    V = np.zeros((m * p, rank_seed))
    W = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_seed
    
    # Try decreasing ranks
    for rank in range(rank_seed - 1, max(rank_seed - 10, 0), -1):
        if time.time() > deadline - 10:
            break
        
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5:
                break
            
            U_try, V_try, W_try, err = try_rank(n, m, p, rank, max_iter=200, tol=1e-9, seed=seed)
            
            if err < 1e-9:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank
                break
        
        if best_rank < rank_seed:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
