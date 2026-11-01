import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if indices are consistent, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute T_hat = sum_r U[:, r] outer V[:, r] outer W[:, r]
        # Efficiently: T_hat[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a tensor contraction.
        # We can use einsum: 'ur,vr,wr->uvw'
        T_hat = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
        return np.allclose(T_hat, T, atol=1e-9)

    # Function to round to half-integers
    def round_half(X):
        return np.round(X * 2) / 2

    # Schoolbook solution as fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1.0
                V_sb[v, col] = 1.0
                W_sb[w, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks from 18 up to 23
    # Known rank for 2x3x4 is likely 18 or 19.
    for R in range(18, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Run Alternating Least Squares (ALS)
            # Fix V, W, solve for U
            # For each u, we have a linear system A_u u_u = b_u
            # A_u has rows indexed by (v, w).
            # A_u[v, w, r] = V[v, r] W[w, r]
            # b_u[v, w] = T[u, v, w]
            
            # Precompute structure for efficiency?
            # Dimensions are small, so explicit loops are fine.
            
            for step in range(50):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # We can vectorize this if we stack the systems?
                # Or just loop over u. n*m is 6.
                for u in range(n*m):
                    # Build A_u (m*p*p*n x R) and b_u (m*p*p*n)
                    # Actually, we can flatten v and w.
                    # A_u is (m*p, p*n, R) -> reshape to (m*p*p*n, R)
                    A_u = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                    b_u = T[u, :, :].reshape(-1)
                    # Solve A_u U_u = b_u
                    # U_u is vector of length R
                    U[u, :], _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                
                # Solve for V
                for v in range(m*p):
                    A_v = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, R)
                    b_v = T[:, v, :].reshape(-1)
                    V[v, :], _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    
                # Solve for W
                for w in range(p*n):
                    A_w = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, R)
                    b_w = T[:, :, w].reshape(-1)
                    W[w, :], _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                
                # Check convergence or round and verify
                if step % 10 == 0:
                    U_r = round_half(U)
                    V_r = round_half(V)
                    W_r = round_half(W)
                    if verify(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
