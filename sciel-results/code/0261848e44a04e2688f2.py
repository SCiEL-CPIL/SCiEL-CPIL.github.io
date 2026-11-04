"""Matrix multiplication 2x5 by 5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the 3-way tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i' else 0."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] = delta_{j,j'}*delta_{k,k'}*delta_{i,i'}"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n))
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_true = _build_tensor(n, m, p)
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < tol

def _make_schoolbook(n, m, p):
    """Create the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin before 150s kill
    
    # Start with schoolbook rank 50
    best_result = _make_schoolbook(n, m, p)
    best_rank = n * m * p  # 50
    
    T = _build_tensor(n, m, p)
    
    # Precompute matricizations
    T_mat1 = T.reshape(n * m, m * p * p * n)       # (10, 250)
    T_mat2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)  # (25, 100)
    T_mat3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)  # (10, 250)
    
    # SVD lower bounds
    s1 = np.linalg.svd(T_mat1, compute_uv=False)
    s2 = np.linalg.svd(T_mat2, compute_uv=False)
    s3 = np.linalg.svd(T_mat3, compute_uv=False)
    tol_svd = 1e-10
    r1 = np.sum(s1 > tol_svd)
    r2 = np.sum(s2 > tol_svd)
    r3 = np.sum(s3 > tol_svd)
    lower_bound = max(r1, r2, r3)
    
    # Try ranks from lower_bound up to best_rank - 1
    for R in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        # Number of attempts based on remaining time
        remaining = deadline - time.time()
        n_attempts = max(5, min(30, int(remaining / 3)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 7919 + 1337)
            
            # Initialize using SVD of mode-1 matricization
            U_init, s_init, Vt_init = np.linalg.svd(T_mat1, full_matrices=False)
            
            if R <= min(U_init.shape[0], Vt_init.shape[0]):
                U = U_init[:, :R].copy()
                # Vt_init[:R] has shape (R, m*p*p*n)
                # Reshape to (R, m*p, p*n)
                Vt_R = Vt_init[:R, :].reshape(R, m * p, p * n)
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    mat = Vt_R[r]  # (m*p, p*n)
                    try:
                        u_mat, s_mat, vt_mat = np.linalg.svd(mat, full_matrices=False)
                        if len(s_mat) > 0 and s_mat[0] > 1e-10:
                            V[:, r] = u_mat[:, 0] * np.sqrt(s_mat[0])
                            W[:, r] = vt_mat[0, :] * np.sqrt(s_mat[0])
                        else:
                            V[:, r] = rng.randn(m * p)
                            W[:, r] = rng.randn(p * n)
                    except:
                        V[:, r] = rng.randn(m * p)
                        W[:, r] = rng.randn(p * n)
            else:
                U = rng.randn(n * m, R)
                V = rng.randn(m * p, R)
                W = rng.randn(p * n, R)
            
            # ALS-style iterative refinement
            max_steps = 200
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Update U
                M_U = np.zeros((m * p * p * n, R))
                for r in range(R):
                    M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                try:
                    M_UM = M_U.T @ M_U
                    if np.linalg.cond(M_UM) < 1e12:
                        U = T_mat1 @ M_U @ np.linalg.inv(M_UM)
                    else:
                        U = T_mat1 @ M_U @ np.linalg.pinv(M_UM)
                except:
                    pass
                
                # Update V
                M_V = np.zeros((n * m * p * n, R))
                for r in range(R):
                    M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                try:
                    M_VM = M_V.T @ M_V
                    if np.linalg.cond(M_VM) < 1e12:
                        V = T_mat2 @ M_V @ np.linalg.inv(M_VM)
                    else:
                        V = T_mat2 @ M_V @ np.linalg.pinv(M_VM)
                except:
                    pass
                
                # Update W
                M_W = np.zeros((n * m * m * p, R))
                for r in range(R):
                    M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                try:
                    M_WM = M_W.T @ M_W
                    if np.linalg.cond(M_WM) < 1e12:
                        W = T_mat3 @ M_W @ np.linalg.inv(M_WM)
                    else:
                        W = T_mat3 @ M_W @ np.linalg.pinv(M_WM)
                except:
                    pass
                
                # Check convergence every 10 steps
                if step % 10 == 9:
                    T_recon = np.zeros((n * m, m * p, p * n))
                    for r in range(R):
                        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    err = np.max(np.abs(T_recon - T))
                    if err < 1e-8:
                        found = True
                        break
            
            if found:
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if found:
            break
    
    return best_result
