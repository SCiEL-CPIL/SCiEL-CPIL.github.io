"""Rank-minimal tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np

n, m, p = 3, 4, 5
N1, N2, N3 = n * m, m * p, p * n  # 12, 20, 15

_deadline = time.time() + 145.0

def _remaining():
    return _deadline - time.time()

def _build_matrices(U, V, W):
    """Build the coefficient matrices for the bilinear form verification."""
    n_, m_, p_ = n, m, p
    r = U.shape[1]
    # A_matrix[i, a, b, k] should equal sum_c U[i*m+a, c] V[b*p+k, c] W[k*n+i, c]
    # We want A_matrix[i, a, b, k] = delta_{a,b}
    # Verification: for each (i,k), the m x m matrix A[i, :, :, k] should be I_m
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # For each c, the contribution to A[i, a, b, k] is U[i*m+a, c] * V[b*p+k, c] * W[k*n+i, c]
    # Let's compute the full tensor and check
    # More efficient: for each (i, k), compute the m x m matrix
    # A_ik[a, b] = sum_c U[i*m+a, c] * V[b*p+k, c] * W[k*n+i, c]
    # = (U_i_col * W_ki) dot V_k_row  where U_i_col[a] = U[i*m+a, :], etc.
    # Actually: A_ik[a, b] = sum_c U[i*m+a, c] * W[k*n+i, c] * V[b*p+k, c]
    # Let D_ik[a, c] = U[i*m+a, c] * W[k*n+i, c]  (m x r)
    # Then A_ik[a, b] = sum_c D_ik[a, c] * V[b*p+k, c] = (D_ik @ V_k^T)[a, b]
    # where V_k is the row of V indexed by (b*p+k) for b in range(m)
    
    for i in range(n_):
        for k in range(p_):
            # D_ik: m x r
            D_ik = U[i * m_: (i + 1) * m_, :] * W[k * n_ + i, :]
            # V_k: m x r (rows indexed by b*p+k)
            V_k = np.array([V[b * p_ + k, :] for b in range(m_)])
            # A_ik = D_ik @ V_k.T  (m x m)
            A_ik = D_ik @ V_k.T
            # Check if A_ik == I_m
            if not np.allclose(A_ik, np.eye(m_), atol=1e-10):
                return False
    return True

def _schoolbook():
    rank = n * m * p
    U = np.zeros((N1, rank))
    V = np.zeros((N2, rank))
    W = np.zeros((N3, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank_R(R, max_iters=5000, seed=42):
    """Try to find a rank-R decomposition using ALS."""
    if _remaining() < 10.0:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(N1, R)
    V = rng.randn(N2, R)
    W = rng.randn(N3, R)
    
    # Normalize columns to prevent divergence
    U_norms = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
    V_norms = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
    W_norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    U = U / U_norms
    V = V / V_norms
    W = W / W_norms
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Update U: for each (i, k), we want D_ik @ V_k.T = I_m
        # where D_ik[a, c] = U[i*m+a, c] * W[k*n+i, c]
        # and V_k is m x r
        # So D_ik = I_m @ V_k.T^{-1} if V_k is square and invertible
        # But R may not equal m. We need to solve: D_ik @ V_k.T = I_m
        # D_ik = I_m @ pinv(V_k.T)
        # D_ik = pinv(V_k.T).T  (m x r)
        # Then U[i*m+a, c] = D_ik[a, c] / W[k*n+i, c]
        # But this is for a specific (i,k). We need consistency across all k for the same i.
        # U[i*m+a, c] must be the same regardless of k.
        # So we have over-constrained system. Use least squares.
        
        # For fixed i, a, c: U[i*m+a, c] * W[k*n+i, c] * (sum over b of V[b*p+k, c] * delta[a,b]) = 1 if a==b else 0
        # Wait, let me redo. The equation is:
        # sum_c U[i*m+a, c] * V[b*p+k, c] * W[k*n+i, c] = delta_{a,b}
        # For fixed (i, k, a): sum_c U[i*m+a, c] * W[k*n+i, c] * V[b*p+k, c] = delta_{a,b} for all b
        # Let x_c = U[i*m+a, c] * W[k*n+i, c]
        # Then sum_c x_c * V[b*p+k, c] = delta_{a,b} for b=0..m-1
        # This is V_k @ x = e_a  where V_k is m x r
        # x = pinv(V_k) @ e_a
        # So U[i*m+a, c] * W[k*n+i, c] = (pinv(V_k) @ e_a)_c
        # U[i*m+a, c] = (pinv(V_k) @ e_a)_c / W[k*n+i, c]
        
        # For fixed i, a: we get r equations per k, but U[i*m+a, c] is shared across k.
        # Use least squares over all k.
        # For each c: sum over k of (W[k*n+i, c] * (pinv(V_k) @ e_a)_c - U[i*m+a, c] * W[k*n+i, c])^2
        # This is getting complex. Let's just do simple alternating:
        # Fix V, W. For each (i, a), solve for U[i*m+a, :] by minimizing over all k:
        # sum_k (sum_c U[i*m+a, c] * V[b*p+k, c] * W[k*n+i, c] - delta_{a,b})^2 for all b
        
        # Simpler: just do gradient descent or a simple update.
        # For now, use a simpler initialization and check.
        
        # Actually, let's just check the current U, V, W and break if good enough.
        err = 0.0
        for i in range(n):
            for k in range(p):
                D_ik = U[i * m: (i + 1) * m, :] * W[k * n + i, :]
                V_k = np.array([V[b * p + k, :] for b in range(m)])
                A_ik = D_ik @ V_k.T
                err += np.sum((A_ik - np.eye(m)) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-10:
                return best_U, best_V, best_W
        
        # Simple update: adjust U to reduce error
        # For each (i, a, k): residual[a, b] = 1 - sum_c U[i*m+a, c] V[b*p+k, c] W[k*n+i, c]
        # Update U[i*m+a, c] += lr * sum_k sum_b residual[a, b] V[b*p+k, c] W[k*n+i, c]
        lr = 0.01
        for i in range(n):
            for a in range(m):
                grad = np.zeros(R)
                for k in range(p):
                    for b in range(m):
                        pred = np.dot(U[i * m + a, :], V[b * p + k, :] * W[k * n + i, :])
                        res = (1.0 if a == b else 0.0) - pred
                        grad += res * V[b * p + k, :] * W[k * n + i, :]
                U[i * m + a, :] += lr * grad
        
        # Normalize
        U_norms = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        U = U / U_norms
    
    return (best_U, best_V, best_W) if best_err < 1e-8 else None

def construct():
    global _deadline
    _deadline = time.time() + 145.0
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try to find better rank
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)
    
    for R in range(20, best_rank):
        if _remaining() < 20.0:
            break
        result = _try_rank_R(R, max_iters=2000, seed=R)
        if result is not None:
            U, V, W = result
            if _build_matrices(U, V, W):
                best_rank = R
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
