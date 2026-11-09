"""Matrix multiplication 2x4x6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 48
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Precompute target tensor
    target = np.zeros((n, m, m, p, p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, j, k, k, i] = 1.0
    target_flat = target.flatten()
    
    def compute_tensor(U, V, W):
        """Compute the full tensor from U, V, W."""
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kpc->ijkkpc', U_3d, V_3d, W_3d)
        return T
    
    def verify(U, V, W):
        """Verify exact decomposition."""
        T = compute_tensor(U, V, W)
        return np.allclose(T, target, atol=1e-6)
    
    def snap_to_halfint(arr):
        """Snap to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    def als_solve(U, V, W, n_iters=80, tol=1e-8):
        """Vectorized alternating least squares."""
        r = U.shape[1]
        
        for it in range(n_iters):
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            W_3d = W.reshape(p, n, r)
            
            # Fix V, W; solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # A[c, (jp,k,kp,ip)] = V[jp,k,c] * W[kp,ip,c]
                    # b[(jp,k,kp,ip)] = target[i,j,jp,k,kp,ip]
                    # Vectorize: A has shape (r, m*p*p*n)
                    # A[:, (jp,k,kp,ip)] = V_3d[jp,k,:] * W_3d[kp,ip,:]
                    
                    # Build A efficiently
                    # For each (jp,k): V_3d[jp,k,:] is shape (r,)
                    # For each (kp,ip): W_3d[kp,ip,:] is shape (r,)
                    # A[:, idx(jp,k,kp,ip)] = V_3d[jp,k,:] * W_3d[kp,ip,:]
                    
                    A = np.zeros((r, m * p * p * n))
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            v_vec = V_3d[jp, k, :]  # shape (r,)
                            for kp in range(p):
                                for ip in range(n):
                                    w_vec = W_3d[kp, ip, :]  # shape (r,)
                                    A[:, idx] = v_vec * w_vec
                                    idx += 1
                    
                    b = target[i, j, :, :, :, :].flatten()
                    
                    # Solve: minimize ||A^T @ x - b|| where x = U[i,j,:]
                    # i.e., (A @ A^T) x = A @ b
                    try:
                        U_new[i, j, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        U_new[i, j, :] = U[i, j, :]
            U = U_new
            
            # Fix U, W; solve for V
            V_new = np.zeros_like(V)
            U_3d = U.reshape(n, m, r)
            for jp in range(m):
                for k in range(p):
                    A = np.zeros((r, n * m * p * n))
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            u_vec = U_3d[i, j, :]
                            for kp in range(p):
                                for ip in range(n):
                                    w_vec = W_3d[kp, ip, :]
                                    A[:, idx] = u_vec * w_vec
                                    idx += 1
                    
                    b = target[:, :, jp, k, :, :].transpose(0, 1, 4, 3, 2).reshape(-1)
                    # Wait, need to be careful with indexing
                    # target[i, j, jp, k, kp, ip] for fixed jp, k
                    # We want b[(i,j,kp,ip)] = target[i, j, jp, k, kp, ip]
                    b = target[:, :, jp, k, :, :].reshape(-1)
                    
                    try:
                        V_new[jp, k, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        V_new[jp, k, :] = V[jp, k, :]
            V = V_new
            
            # Fix U, V; solve for W
            W_new = np.zeros_like(W)
            V_3d = V.reshape(m, p, r)
            for kp in range(p):
                for ip in range(n):
                    A = np.zeros((r, n * m * m * p))
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            u_vec = U_3d[i, j, :]
                            for jp in range(m):
                                for k in range(p):
                                    v_vec = V_3d[jp, k, :]
                                    A[:, idx] = u_vec * v_vec
                                    idx += 1
                    
                    # target[i, j, jp, k, kp, ip] for fixed kp, ip
                    b = target[:, :, :, :, kp, ip].reshape(-1)
                    
                    try:
                        W_new[kp, ip, :] = np.linalg.lstsq(A.T, b, rcond=None)[0]
                    except:
                        W_new[kp, ip, :] = W[kp, ip, :]
            W = W_new
            
            # Check convergence
            T = compute_tensor(U, V, W)
            residual = np.linalg.norm(T - target)
            if residual < tol:
                break
        
        return U, V, W, residual
    
    # Try ranks from low to high
    for r in range(24, 48):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 8
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(98765 + restart * 31 + r * 17)
            
            # Initialize
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            # Run ALS
            U, V, W, residual = als_solve(U, V, W, n_iters=40)
            
            # Snap to half-integers
            U_snapped = snap_to_halfint(U)
            V_snapped = snap_to_halfint(V)
            W_snapped = snap_to_halfint(W)
            
            # Verify
            if verify(U_snapped, V_snapped, W_snapped):
                best_rank = r
                best_result = {
                    "U": U_snapped.tolist(),
                    "V": V_snapped.tolist(),
                    "W": W_snapped.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    return best_result
