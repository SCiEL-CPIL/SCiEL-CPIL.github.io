import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Build the target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U,V,W exactly decompose T. Returns True if error < 1e-9."""
        Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.sum((Res - T) ** 2)
        return err < 1e-9, err

    def snap_to_half_integers(U, V, W):
        """Snap values to nearest half-integer and verify."""
        bU = np.round(U * 2) / 2
        bV = np.round(V * 2) / 2
        bW = np.round(W * 2) / 2
        ok, err = verify(bU, bV, bW)
        if ok:
            return bU, bV, bW, True
        return None, None, None, False

    def als_decompose(T, rank, max_iter=200, tol=1e-10, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        np.random.seed(seed)
        n_u, n_v, n_w = T.shape
        
        # Initialize with random values
        U = np.random.randn(n_u, rank) * 0.5
        V = np.random.randn(n_v, rank) * 0.5
        W = np.random.randn(n_w, rank) * 0.5
        
        prev_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each a, solve linear system
            # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            # Reshape: T[a].flatten() = sum_r U[a,r] * (V.flatten() * W.flatten())
            # So: T_flat[a] = U[a,:] @ (V.flatten() * W.flatten())
            # This is: U[a,:] should satisfy U[a,:] @ M = T[a].flatten()
            # where M[r] = sum_{b,c} V[b,r]*W[c,r] ... no that's wrong
            
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:].flatten() = sum_r U[a,r] * (V[:,r] * W[:,r]).reshape(-1)
            # Let v_w[r] = (V[:,r] * W[:,r]).reshape(-1)  -- this is a vector of length n_v*n_w
            # Then T[a].flatten() = sum_r U[a,r] * v_w[r]
            # So: U[a,:] @ V_W_matrix = T[a].flatten()
            # where V_W_matrix[r, b*n_w+c] = V[b,r] * W[c,r]
            
            # Build V_W matrix: (rank, n_v*n_w)
            V_W = np.zeros((rank, n_v * n_w))
            for r in range(rank):
                V_W[r, :] = (V[:, r] * W[:, r]).reshape(-1)
            
            # For each a: solve U[a,:] @ V_W = T[a].flatten()
            # This is: V_W.T @ U[a,:] = T[a].flatten()
            try:
                U_new, _, _, _ = np.linalg.lstsq(V_W.T, T.reshape(n_u, -1).T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V: for each b
            U_W = np.zeros((rank, n_u * n_w))
            for r in range(rank):
                U_W[r, :] = (U[:, r] * W[:, r]).reshape(-1)
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(U_W.T, T.transpose(1, 0, 2).reshape(n_v, -1).T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W: for each c
            U_V = np.zeros((rank, n_u * n_v))
            for r in range(rank):
                U_V[r, :] = (U[:, r] * V[:, r]).reshape(-1)
            
            try:
                W_new, _, _, _ = np.linalg.lstsq(U_V.T, T.transpose(2, 0, 1).reshape(n_w, -1).T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Compute error
            Res = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.sum((Res - T) ** 2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err

    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Known lower bound is max(n*m, m*p, p*n) = 27
    # Try ranks from 27 upward
    for rank in range(27, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            
            U, V, W, err = als_decompose(T, rank, max_iter=100, seed=seed)
            
            if err < 1e-9:
                # Snap to half-integers
                bU, bV, bW, ok = snap_to_half_integers(U, V, W)
                if ok:
                    return {"U": bU.tolist(), "V": bV.tolist(), "W": bW.tolist()}
                # If snap fails, check if it's close enough with rational approximation
                # Try with more iterations
                U2, V2, W2, err2 = als_decompose(T, rank, max_iter=500, seed=seed)
                if err2 < 1e-9:
                    bU2, bV2, bW2, ok2 = snap_to_half_integers(U2, V2, W2)
                    if ok2:
                        return {"U": bU2.tolist(), "V": bV2.tolist(), "W": bW2.tolist()}
        
        # If we get close but not exact, try polishing
        if err < 1e-6:
            # Try to find exact solution with more iterations
            U, V, W, err = als_decompose(T, rank, max_iter=1000, tol=1e-12, seed=42)
            if err < 1e-9:
                bU, bV, bW, ok = snap_to_half_integers(U, V, W)
                if ok:
                    return {"U": bU.tolist(), "V": bV.tolist(), "W": bW.tolist()}

    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
