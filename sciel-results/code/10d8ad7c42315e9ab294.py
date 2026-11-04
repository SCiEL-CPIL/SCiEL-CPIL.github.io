"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _matmul_tensor(n, m, p):
    """Build the flattened matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W decompose the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _matmul_tensor(n, m, p)
    for r in range(U.shape[1]):
        T -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(T)) < tol

def _try_rank_greedy(n, m, p, rank, deadline, seed=42):
    """Try to find a rank decomposition via random initialization + alternating least squares."""
    rng = np.random.default_rng(seed)
    T = _matmul_tensor(n, m, p)
    
    # Try several random initializations
    for trial in range(50):
        if time.time() > deadline:
            return None
        U = rng.normal(size=(n * m, rank))
        V = rng.normal(size=(m * p, rank))
        W = rng.normal(size=(p * n, rank))
        
        # Normalize columns
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:, r]**2) + np.sum(V[:, r]**2) + np.sum(W[:, r]**2))
            if norm > 1e-12:
                U[:, r] /= norm
                V[:, r] /= norm
                W[:, r] /= norm
        
        # Alternating least squares
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Update U: for each r, solve min ||T - sum_r (U_r ⊗ V_r ⊗ W_r)||
            # U[:, r] = sum_{j,k} V[j,r]*W[k,r] * T[:, j, k] ... not quite
            # Better: unfold T into a matrix.
            # T reshaped: for fixed (j,k), T[:, j, k] is a vector of length n*m
            # The contribution of component r to T[:, j, k] is U[:, r] * V[j, r] * W[k, r]
            # So T = sum_r (U[:, r] * V[j, r] * W[k, r]) over j,k
            # This is: T_ikj = sum_r U[i,r] * V[j,r] * W[k,r]
            # Unfold along i: T_i(j,k,r) = sum_r U[i,r] * (V[j,r]*W[k,r])
            # So for fixed i: T_i = sum_r U[i,r] * (V_jk_r) where V_jk_r[j,k] = V[j,r]*W[k,r]
            
            # Let's use a different approach: treat as tensor train / ALS
            # Reshape T to (n*m, m*p, p*n)
            # For ALS, fix V and W, solve for U:
            # T_ikj = sum_r U_ir * V_jr * W_kr
            # For each i: T_i(j,k) = sum_r U_ir * (V_r(j) * W_r(k))
            # This is a linear system in U_i for each r... but it's not standard LS.
            
            # Actually, let's use the standard approach for tensor decomposition:
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # Unfold T along mode 1: T_(1) is a matrix of shape (n*m, (m*p)*(p*n))
            # T_(1)[i, (j,k)] = sum_r U[i,r] * V[j,r] * W[k,r]
            # = sum_r U[i,r] * (V[j,r] * W[k,r])
            # So T_(1) = U * (V ⊗ W)^T where (V ⊗ W) is the Khatri-Rao product
            # Khatri-Rao of V and W: shape (m*p*p*n, rank) -> no, it's (m*p, rank) and (p*n, rank)
            # Kronecker product per row: for each column r, V[:,r] ⊗ W[:,r] has shape (m*p*p*n,)
            # So T_(1) = U * (V ⊗ W)^T where V ⊗ W has shape ((m*p)*(p*n), rank)
            
            # Let's build the Khatri-Rao product
            # For each r, V[:,r] ⊗ W[:,r] is a vector of length (m*p)*(p*n)
            # The Khatri-Rao matrix K has shape ((m*p)*(p*n), rank)
            # K[(j*p+k) * (p*n) + (k*n+i)... no, this is getting complicated.
            
            # Let me use a simpler verification and a different optimization.
            # Since the dimensions are small, let's just use scipy.optimize or manual gradients.
            # But that might be slow.
            
            # Alternative: use the fact that for small dimensions, we can try known constructions.
            # For 4x5x6, the schoolbook rank is 120. 
            # There's a known bound: rank of matmul tensor for n x m x p is at most n*m*p.
            # For small cases, sometimes you can do better.
            # But I don't have a specific construction for 4x5x6.
            
            # Let's try a different approach: use scipy to solve for U given V,W or vice versa.
            # Fix V and W, solve for U:
            # T_ikj = sum_r U_ir * V_jr * W_kr
            # For fixed j,k: T_ikj = sum_r U_ir * (V_jr * W_kr)
            # This is: T[:, j, k] = sum_r U[:, r] * (V[j,r] * W[k,r])
            # So T[:, j, k] = U * (V[j,:] * W[k,:])^T
            # Stack over (j,k): T_unfolded = U * K^T where K[(j,k), r] = V[j,r] * W[k,r]
            # K shape: (m*p*p*n, rank) -- no, K shape is (m*p, p*n, rank) reshaped to (m*p*p*n, rank)
            
            # Build K: K[(j*p+k), (k*n+i)... no, K is indexed by (j, k) and r
            # K has shape (m*p, p*n, rank) -> reshape to (m*p*p*n, rank)
            # But T unrolled along i has shape (n*m, m*p*p*n)
            # T_unfolded[i, (j, k)] = T[i, j, k] where i in [0, n*m), j in [0, m*p), k in [0, p*n)
            # Wait, T has shape (n*m, m*p, p*n). Unfolding along mode 0 gives (n*m, (m*p)*(p*n))
            # T_unfolded[i, idx] where idx corresponds to (j, k) in mode 1 and mode 2
            # idx = j * (p*n) + k
            # T_unfolded[i, j*(p*n)+k] = T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # = sum_r U[i,r] * (V[j,r] * W[k,r])
            # So T_unfolded = U * (V ⊙ W)^T where (V ⊙ W)[(j,k), r] = V[j,r] * W[k,r]
            # (V ⊙ W) has shape (m*p, p*n, rank) -> flattened to (m*p*p*n, rank)
            
            # Build the Khatri-Rao product
            K = np.zeros((m * p * p * n, rank))
            for j in range(m * p):
                for k in range(p * n):
                    idx = j * (p * n) + k
                    K[idx, :] = V[j, :] * W[k, :]
            
            # Now T_unfolded = U * K^T
            # So U = T_unfolded * pinv(K^T) = T_unfolded * pinv(K).T
            T_unfolded = T.reshape(n * m, m * p * p * n)
            try:
                U_new = T_unfolded @ np.linalg.pinv(K.T)
            except:
                continue
            U = U_new
            
            # Similarly for V
            K_v = np.zeros((n * m * p * n, rank))
            for i in range(n * m):
                for k in range(p * n):
                    idx = i * (p * n) + k
                    K_v[idx, :] = U[i, :] * W[k, :]
            T_unfolded_v = T.reshape(m * p, n * m * p * n)
            T_unfolded_v = np.transpose(T, (1, 0, 2)).reshape(m * p, n * m * p * n)
            try:
                V_new = T_unfolded_v @ np.linalg.pinv(K_v.T)
            except:
                continue
            V = V_new
            
            # Similarly for W
            K_w = np.zeros((n * m * m * p, rank))
            for i in range(n * m):
                for j in range(m * p):
                    idx = i * (m * p) + j
                    K_w[idx, :] = U[i, :] * V[j, :]
            T_unfolded_w = T.reshape(p * n, n * m * m * p)
            T_unfolded_w = np.transpose(T, (2, 0, 1)).reshape(p * n, n * m * m * p)
            try:
                W_new = T_unfolded_w @ np.linalg.pinv(K_w.T)
            except:
                continue
            W = W_new
        
        # Verify
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return [U.tolist(), V.tolist(), W.tolist()]
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10 second safety margin
    
    # First, try to find a lower rank decomposition
    schoolbook_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks from lower to higher
    for rank in range(100, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        result = _try_rank_greedy(n, m, p, rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    # If no better rank found, use schoolbook
    if best_U is None:
        rank = schoolbook_rank
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
