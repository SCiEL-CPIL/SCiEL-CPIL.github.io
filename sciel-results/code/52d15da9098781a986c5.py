import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution as fallback
    def schoolbook():
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
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_half_int(arr):
        """Round to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    def check_exact(U, V, W, T, tol=1e-9):
        """Check if sum_c U[:,c] \otimes V[:,c] \otimes W[:,c] == T"""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=tol)
    
    def try_rank(r, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            VW = np.einsum('bd,cd->bcd', V, W)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            try:
                U = T_2d @ np.linalg.pinv(VW_2d.T)
            except:
                pass
            
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V = T_2d_v @ np.linalg.pinv(UW_2d.T)
            except:
                pass
            
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W = T_2d_w @ np.linalg.pinv(UV_2d.T)
            except:
                pass
            
            # Check error
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Try rounding to half-integers
        U_h = to_half_int(best_U)
        V_h = to_half_int(best_V)
        W_h = to_half_int(best_W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
        
        return None
    
    best_result = None
    # Try ranks from 50 down to 15
    for r in range(50, 14, -1):
        if time.time() > deadline - 10:
            break
        
        found = None
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=50, seed=seed + r)
            if result is not None:
                found = result
                break
        
        if found is not None:
            U, V, W = found
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U_list = [[float(x) for x in row] for row in U_sb]
    V_list = [[float(x) for x in row] for row in V_sb]
    W_list = [[float(x) for x in row] for row in W_sb]
    return {"U": U_list, "V": V_list, "W": W_list}
