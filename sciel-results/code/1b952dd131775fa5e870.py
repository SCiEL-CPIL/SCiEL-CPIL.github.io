import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    I, J, K = T.shape
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.array_equal(reconstructed, T)

def _try_rank_decomposition(T, rank, max_iterations=50, n_restarts=10, deadline=None):
    """Try to find a rank-r CP decomposition of T using ALS with integer snapping."""
    I, J, K = T.shape
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with random small integers
        np.random.seed(42 + restart)
        U = np.random.randint(-2, 3, size=(I, rank))
        V = np.random.randint(-2, 3, size=(J, rank))
        W = np.random.randint(-2, 3, size=(K, rank))
        
        # Make sure at least some entries are non-zero
        if np.all(U == 0):
            U[0, 0] = 1
        if np.all(V == 0):
            V[0, 0] = 1
        if np.all(W == 0):
            W[0, 0] = 1
        
        for iteration in range(max_iterations):
            if deadline and time.time() > deadline:
                break
            
            # Update U
            for i in range(I):
                # Build design matrix: D[j*K + k, c] = V[j, c] * W[k, c]
                V_rep = V[:, np.newaxis, :]
                W_rep = W[np.newaxis, :, :]
                Prod = V_rep * W_rep
                D = Prod.reshape(J * K, rank)
                
                target = T[i].flatten()
                
                # Solve D.T @ u = target
                try:
                    u_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                    U[i, :] = np.round(u_sol)
                except:
                    pass
            
            # Update V
            for j in range(J):
                U_rep = U[:, np.newaxis, :]
                W_rep = W[np.newaxis, :, :]
                Prod = U_rep * W_rep
                D = Prod.reshape(I * K, rank)
                
                target = T[:, j, :].flatten()
                
                try:
                    v_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                    V[j, :] = np.round(v_sol)
                except:
                    pass
            
            # Update W
            for k in range(K):
                U_rep = U[:, np.newaxis, :]
                V_rep = V[np.newaxis, :, :]
                Prod = U_rep * V_rep
                D = Prod.reshape(I * J, rank)
                
                target = T[:, :, k].flatten()
                
                try:
                    w_sol, residuals, rank_sol, s = np.linalg.lstsq(D.T, target, rcond=None)
                    W[k, :] = np.round(w_sol)
                except:
                    pass
            
            # Check exactness
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.array_equal(T_recon, T):
                return U, V, W
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    I, J, K = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((I, rank_schoolbook), dtype=int)
    V_school = np.zeros((J, rank_schoolbook), dtype=int)
    W_school = np.zeros((K, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try lower ranks from 47 down to 30
    for rank in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank_decomposition(T, rank, max_iterations=30, n_restarts=5, deadline=deadline)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            break  # Found a valid decomposition, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
