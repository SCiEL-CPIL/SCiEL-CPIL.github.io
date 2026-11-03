import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()

def _check_time(deadline):
    return time.time() > deadline

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using multiple test matrices."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # Use multiple random test matrices for robustness
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C_computed using vectorized operations
        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            # term_U[i] = sum_j U[i*m+j, r] * A[i,j]
            U_row = U_arr[:, r].reshape(n, m)
            term_U = (U_row * A).sum(axis=1)
            
            # term_V[k] = sum_j V[j*p+k, r] * B[j,k]
            V_col = V_arr[:, r].reshape(m, p)
            term_V = (V_col * B).sum(axis=0)
            
            # W[k*n+i, r] * term_U[i] * term_V[k]
            W_mat = W_arr[:, r].reshape(p, n)
            C_computed += W_mat.T * np.outer(term_U, term_V)
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _als_decomposition(n, m, p, target_rank, max_iters=500, tol=1e-12, seed=0):
    """Alternating least squares to find rank-target_rank decomposition.
    
    Uses vectorized operations for speed.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    reg = 1e-8 * np.eye(target_rank)
    
    for iteration in range(max_iters):
        # Update U: for each (i,j), solve (V_j^T * W_i^T)(V_j^T * W_i^T) u = V_j^T * W_i^T * ones
        for i in range(n):
            for j in range(m):
                # A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                V_sub = V_arr = V[j * p:(j+1) * p, :]  # (p, rank)
                W_sub = W[k * n + i] if False else W[i::n, :]  # This is wrong, let me fix
                
                # Correct indexing: W[k*n+i, c] for k in range(p)
                W_sub = np.array([W[k * n + i, :] for k in range(p)])  # (p, rank)
                
                A_mat = V_sub * W_sub  # (p, rank)
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # A_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                U_sub = np.array([U[i * m + j, :] for i in range(n)])  # (n, rank)
                W_sub = np.array([W[k * n + i, :] for i in range(n)])  # (n, rank)
                
                A_mat = U_sub * W_sub  # (n, rank)
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # A_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                U_sub = np.array([U[i * m + j, :] for j in range(m)])  # (m, rank)
                V_sub = np.array([V[j * p + k, :] for j in range(m)])  # (m, rank)
                
                A_mat = U_sub * V_sub  # (m, rank)
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence using tensor contraction
        # tensor[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (1.0 - val) ** 2
        
        if error < tol:
            return U.tolist(), V.tolist(), W.tolist(), True
    
    return U.tolist(), V.tolist(), W.tolist(), False

def _polish_decomposition(U_list, V_list, W_list, n, m, p, max_iters=100, tol=1e-14):
    """High-precision polish pass with tighter tolerance."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    target_rank = U.shape[1]
    
    reg = 1e-12 * np.eye(target_rank)
    
    for iteration in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                V_sub = V[j * p:(j+1) * p, :]
                W_sub = np.array([W[k * n + i, :] for k in range(p)])
                A_mat = V_sub * W_sub
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(p)
                    U[i * m + j, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_sub = np.array([U[i * m + j, :] for i in range(n)])
                W_sub = np.array([W[k * n + i, :] for i in range(n)])
                A_mat = U_sub * W_sub
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(n)
                    V[j * p + k, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_sub = np.array([U[i * m + j, :] for j in range(m)])
                V_sub = np.array([V[j * p + k, :] for j in range(m)])
                A_mat = U_sub * V_sub
                try:
                    ATA = A_mat.T @ A_mat + reg
                    ATb = A_mat.T @ np.ones(m)
                    W[k * n + i, :] = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    error += (1.0 - val) ** 2
        
        if error < tol:
            return U.tolist(), V.tolist(), W.tolist(), True
    
    return U.tolist(), V.tolist(), W.tolist(), False

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 115
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds suggest rank >= 42 for 3x4x5, try from 42 upward
    for target_rank in range(42, 61):
        if _check_time(deadline - 20):
            break
        
        found = False
        for seed in range(15):
            if _check_time(deadline - 15):
                break
            
            U_list, V_list, W_list, converged = _als_decomposition(
                n, m, p, target_rank, max_iters=300, tol=1e-10, seed=seed
            )
            
            if converged:
                # Polish with tighter tolerance
                U_polished, V_polished, W_polished, polished_ok = _polish_decomposition(
                    U_list, V_list, W_list, n, m, p, max_iters=50, tol=1e-14
                )
                
                if polished_ok and _verify_decomposition(U_polished, V_polished, W_polished, n, m, p):
                    best = {"U": U_polished, "V": V_polished, "W": W_polished}
                    best_rank = target_rank
                    found = True
                    break
                elif _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best
