import time
import numpy as np
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(mat * 2) / 2
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    # Try integers
    snapped_int = np.round(mat)
    if np.max(np.abs(mat - snapped_int)) < tol:
        return snapped_int
    return mat

def _try_als(rank, n, m, p, max_iter=200, seed=42, time_limit=10):
    """Try alternating least squares to find rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(dim_U, rank) * 0.5
    V = rng.randn(dim_V, rank) * 0.5
    W = rng.randn(dim_W, rank) * 0.5
    
    # Target tensor
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        # Update U: for each a, solve min ||T_target[a,:,:] - sum_r U[a,r]*V[:,r]*W[:,r]||
        # This is: T_target[a,:,:] = U[a,:].reshape(1,rank) @ (V * W) where (V*W)[r,:] = V[:,r]*W[:,r]
        # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Fix V,W: for each a, we have T_target[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
        # So define M_VW[b,c] = sum_r V[b,r]*W[c,r]... no, that's not right.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V,W: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # So we can write: T_target.reshape(dim_U, -1) ≈ U @ (something)
        # Let's define: for each r, let S_r = outer(V[:,r], W[:,r]) which is dim_V x dim_W
        # Then T_target[a,b,c] = sum_r U[a,r] * S_r[b,c]
        # Stack S_r: S = np.stack([outer(V[:,r], W[:,r]) for r in range(rank)])
        # Then T_target[a,:,:] = sum_r U[a,r] * S[r,:,:]
        # This is a linear system: T_target.reshape(dim_U, dim_V*dim_W) = U @ S_reshaped
        # where S_reshaped[r, b*dim_W+c] = S[r,b,c]
        
        S_reshaped = np.zeros((rank, dim_V * dim_W))
        for r in range(rank):
            S_reshaped[r, :] = np.outer(V[:, r], W[:, r]).flatten()
        
        # Solve: T_target.reshape(dim_U, -1) = U @ S_reshaped
        # U = T_target.reshape(dim_U, -1) @ pinv(S_reshaped)
        try:
            U_new = T_target.reshape(dim_U, -1) @ np.linalg.pinv(S_reshaped)
        except:
            break
        
        # Update V
        S_UW = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new = T_target.reshape(dim_V, -1, dim_W).transpose(1, 0, 2).reshape(dim_V * dim_W, dim_U).T @ np.linalg.pinv(S_UW)
            # Actually: T_target[b,a,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # For fixed U,W: T_target[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
            # T_target.reshape(-1, dim_V, dim_W) -> for each b: T_target[:,b,:] = sum_r V[b,r] * S_r
            # Where S_r = outer(U[:,r], W[:,r])
            # So: T_target.transpose(1,0,2).reshape(dim_V, dim_U*dim_W) = V @ S_UW_reshaped
        except:
            break
        
        S_UW_reshaped = np.zeros((rank, dim_U * dim_W))
        for r in range(rank):
            S_UW_reshaped[r, :] = np.outer(U[:, r], W[:, r]).flatten()
        try:
            V_new = T_target.transpose(1, 0, 2).reshape(dim_V, dim_U * dim_W) @ np.linalg.pinv(S_UW_reshaped)
        except:
            break
        
        # Update W
        S_UV_reshaped = np.zeros((rank, dim_U * dim_V))
        for r in range(rank):
            S_UV_reshaped[r, :] = np.outer(U[:, r], V[:, r]).flatten()
        try:
            W_new = T_target.transpose(2, 0, 1).reshape(dim_W, dim_U * dim_V) @ np.linalg.pinv(S_UV_reshaped)
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T_target))
        if err < 1e-10:
            break
    
    # Try snapping
    U_snap = _snap_to_half_integers(U)
    V_snap = _snap_to_half_integers(V)
    W_snap = _snap_to_half_integers(W)
    
    if _verify(U_snap, V_snap, W_snap, n, m, p):
        return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist())
    
    # Also try without snapping
    if _verify(U, V, W, n, m, p):
        return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try ranks from 13 to 23
    # 13 is the known optimal rank for 2x3x4 matrix multiplication
    for rank in range(13, 24):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try ALS with multiple seeds
        max_seeds = 20 if rank >= 15 else 10
        for seed in range(rank * 100, rank * 100 + max_seeds):
            if time.time() > _DEADLINE - 3:
                break
            res = _try_als(rank, n, m, p, max_iter=100, seed=seed, time_limit=3)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
