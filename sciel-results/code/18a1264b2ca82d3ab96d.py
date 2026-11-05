import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    def build_rank44():
        rank = 44
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col = 0
        def add_strassen(rA, cA, rB, cB):
            nonlocal col
            # M1 = (X11+X22)(Y11+Y22)
            U[(rA+0)*m + (cA+0), col] = 1
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+0)*p + (cB+0), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+0)*n + (rA+0), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M2 = (X21+X22)Y11
            U[(rA+1)*m + (cA+0), col] = 1
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+0)*p + (cB+0), col] = 1
            W[(cB+0)*n + (rA+1), col] = 1
            W[(cB+1)*n + (rA+1), col] = -1
            col += 1
            # M3 = X11(Y12-Y22)
            U[(rA+0)*m + (cA+0), col] = 1
            V[(rB+0)*p + (cB+1), col] = 1
            V[(rB+1)*p + (cB+1), col] = -1
            W[(cB+1)*n + (rA+0), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M4 = X22(Y21-Y11)
            U[(rA+1)*m + (cA+1), col] = 1
            V[(rB+1)*p + (cB+0), col] = 1
            V[(rB+0)*p + (cB+0), col] = -1
            W[(cB+0)*n + (rA+0), col] = 1
            W[(cB+0)*n + (rA+1), col] = 1
            col += 1
            # M5 = (X11+X12)Y22
            U[(rA+0)*m + (cA+0), col] = 1
            U[(rA+0)*m + (cA+1), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+1)*n + (rA+0), col] = 1
            W[(cB+0)*n + (rA+0), col] = -1
            col += 1
            # M6 = (X21-X11)(Y11+Y12)
            U[(rA+1)*m + (cA+0), col] = 1
            U[(rA+0)*m + (cA+0), col] = -1
            V[(rB+0)*p + (cB+0), col] = 1
            V[(rB+0)*p + (cB+1), col] = 1
            W[(cB+1)*n + (rA+1), col] = 1
            col += 1
            # M7 = (X12-X22)(Y21+Y22)
            U[(rA+0)*m + (cA+1), col] = 1
            U[(rA+1)*m + (cA+1), col] = -1
            V[(rB+1)*p + (cB+0), col] = 1
            V[(rB+1)*p + (cB+1), col] = 1
            W[(cB+0)*n + (rA+0), col] = 1
            col += 1
        
        # Four 2x2x2 Strassen blocks
        add_strassen(0, 0, 0, 0)
        add_strassen(0, 2, 2, 0)
        add_strassen(0, 0, 0, 2)
        add_strassen(0, 2, 2, 2)
        
        # 1x4x4 schoolbook for row 2
        for j in range(4):
            for k in range(4):
                U[2*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + 2, col] = 1
                col += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    def verify(U, V, W, n, m, p, tol=1e-6):
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol, rtol=0)
    
    if not verify(U44, V44, W44, n, m, p):
        # Fallback to schoolbook
        rank = 48
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    target = np.ones((n, m, p))
    
    # Precompute index arrays for vectorized ALS
    # For updating U[i,j,r]: sum over k of target[i,j,k] * V[j,k,r] * W[k,i,r]
    # We'll build lookup structures
    
    def als_decompose_vec(target, n, m, p, rank, init_U=None, init_V=None, init_W=None, max_iter=200, tol=1e-8):
        """Vectorized ALS decomposition."""
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            np.random.seed(42)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U: for each r, U[i*m+j, r] = sum_k target[i,j,k]*V[j*p+k,r]*W[k*n+i,r] / sum_k V[j*p+k,r]^2*W[k*n+i,r]^2
            for r in range(rank):
                V_r = V[:, r]  # shape (m*p,)
                W_r = W[:, r]  # shape (p*n,)
                
                # Build matrices for efficient computation
                # V_mat[i,j,k] = V[j*p+k, r] for all i
                # W_mat[i,j,k] = W[k*n+i, r] for all j
                V_mat = V_r.reshape(m, p)[None, :, :]  # (1, m, p)
                V_mat = np.broadcast_to(V_mat, (n, m, p))  # (n, m, p)
                
                W_mat = W_r.reshape(p, n)[:, None, :]  # (p, 1, n) -> need (n, m, p)
                # W[k*n+i] -> W_mat[i, j, k] = W_r[k*n+i]
                W_mat = np.zeros((n, m, p))
                for i in range(n):
                    for k in range(p):
                        W_mat[i, :, k] = W_r[k*n+i]
                
                num = target * V_mat * W_mat  # (n, m, p)
                den = V_mat**2 * W_mat**2  # (n, m, p)
                
                # Sum over k (axis=2)
                num_sum = num.sum(axis=2)  # (n, m)
                den_sum = den.sum(axis=2)  # (n, m)
                
                mask = den_sum > 1e-12
                U_flat = U[:, r].reshape(n, m)
                U_flat[mask] = num_sum[mask] / den_sum[mask]
                U[:, r] = U_flat.flatten()
            
            # Update V: for each r, V[j*p+k, r] = sum_i target[i,j,k]*U[i*m+j,r]*W[k*n+i,r] / sum_i U[i*m+j,r]^2*W[k*n+i,r]^2
            for r in range(rank):
                U_r = U[:, r]  # shape (n*m,)
                W_r = W[:, r]  # shape (p*n,)
                
                # U_mat[i,j,k] = U_r[i*m+j]
                U_mat = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        U_mat[i, j, :] = U_r[i*m+j]
                
                # W_mat[i,j,k] = W_r[k*n+i]
                W_mat = np.zeros((n, m, p))
                for i in range(n):
                    for k in range(p):
                        W_mat[i, :, k] = W_r[k*n+i]
                
                num = target * U_mat * W_mat
                den = U_mat**2 * W_mat**2
                
                num_sum = num.sum(axis=0)  # (m, p)
                den_sum = den.sum(axis=0)  # (m, p)
                
                mask = den_sum > 1e-12
                V_flat = V[:, r].reshape(m, p)
                V_flat[mask] = num_sum[mask] / den_sum[mask]
                V[:, r] = V_flat.flatten()
            
            # Update W: for each r, W[k*n+i, r] = sum_j target[i,j,k]*U[i*m+j,r]*V[j*p+k,r] / sum_j U[i*m+j,r]^2*V[j*p+k,r]^2
            for r in range(rank):
                U_r = U[:, r]  # shape (n*m,)
                V_r = V[:, r]  # shape (m*p,)
                
                # U_mat[i,j,k] = U_r[i*m+j]
                U_mat = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        U_mat[i, j, :] = U_r[i*m+j]
                
                # V_mat[i,j,k] = V_r[j*p+k]
                V_mat = np.zeros((n, m, p))
                for j in range(m):
                    for k in range(p):
                        V_mat[:, j, k] = V_r[j*p+k]
                
                num = target * U_mat * V_mat
                den = U_mat**2 * V_mat**2
                
                num_sum = num.sum(axis=1)  # (n, p)
                den_sum = den.sum(axis=1)  # (n, p)
                
                mask = den_sum > 1e-12
                W_flat = W[:, r].reshape(p, n).T  # (n, p)
                W_flat[mask] = num_sum[mask] / den_sum[mask]
                W[:, r] = W_flat.T.flatten()
            
            # Check convergence every 10 iterations
            if it > 0 and it % 10 == 0:
                T = np.zeros((n, m, p))
                for r in range(rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    T += np.einsum('ij,jk,ki->ijk', u, v, w)
                if np.allclose(T, target, atol=tol, rtol=0):
                    break
        
        return U, V, W
    
    def polish(U, V, W, target, n, m, p, max_iter=50, tol=1e-10):
        """High-precision polish pass."""
        rank = U.shape[1]
        for it in range(max_iter):
            if time.time() > deadline:
                break
            U, V, W = als_decompose_vec(target, n, m, p, rank, U, V, W, max_iter=1, tol=tol)
            
            if it > 0 and it % 5 == 0:
                T = np.zeros((n, m, p))
                for r in range(rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    T += np.einsum('ij,jk,ki->ijk', u, v, w)
                if np.allclose(T, target, atol=tol, rtol=0):
                    break
        
        return U, V, W
    
    # Try to reduce rank
    for target_rank in [43, 42, 41, 40]:
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # For rank 43: try dropping single terms
        if target_rank == 43:
            for drop_idx in range(44):
                if time.time() > deadline - 30:
                    break
                U_init = np.delete(U44, drop_idx, axis=1)
                V_init = np.delete(V44, drop_idx, axis=1)
                W_init = np.delete(W44, drop_idx, axis=1)
                
                U_new, V_new, W_new = als_decompose_vec(target, n, m, p, target_rank, U_init, V_init, W_init, max_iter=150)
                U_new, V_new, W_new = polish(U_new, V_new, W_new, target, n, m, p, max_iter=30)
                
                if verify(U_new, V_new, W_new, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break
        
        # For rank 42: try dropping pairs of terms
        elif target_rank == 42:
            # Try all pairs of drops (combinations of 2 from 44 = 946 pairs, too many)
            # Instead, try dropping adjacent terms or terms from same Strassen block
            for drop1 in range(44):
                if time.time() > deadline - 30:
                    break
                for drop2 in range(drop1+1, 44):
                    if time.time() > deadline - 30:
                        break
                    # Skip if too many combinations already
                    if drop1 > 20:
                        break
                    if drop2 > 20:
                        break
                    
                    idx = np.delete(np.arange(44), [drop1, drop2])
                    U_init = U44[:, idx]
                    V_init = V44[:, idx]
                    W_init = W44[:, idx]
                    
                    U_new, V_new, W_new = als_decompose_vec(target, n, m, p, target_rank, U_init, V_init, W_init, max_iter=150)
                    U_new, V_new, W_new = polish(U_new, V_new, W_new, target, n, m, p, max_iter=30)
                    
                    if verify(U_new, V_new, W_new, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                        found = True
                        break
                if found:
                    break
        
        # For rank 41, 40: random restarts
        else:
            np.random.seed(target_rank)
            for trial in range(5):
                if time.time() > deadline - 30:
                    break
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                
                U_new, V_new, W_new = als_decompose_vec(target, n, m, p, target_rank, U_init, V_init, W_init, max_iter=300)
                U_new, V_new, W_new = polish(U_new, V_new, W_new, target, n, m, p, max_iter=30)
                
                if verify(U_new, V_new, W_new, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break
        
        if found:
            break
    
    # Final polish on the best result
    if best_rank < 44:
        best_U, best_V, best_W = polish(best_U, best_V, best_W, target, n, m, p, max_iter=50, tol=1e-12)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
