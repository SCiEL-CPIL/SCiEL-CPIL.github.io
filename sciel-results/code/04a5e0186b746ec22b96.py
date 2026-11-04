import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Tensor T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[:,r] V[:,r] W[:,r]^T == T
        # Efficient check:
        # For each r, add outer product
        # But T is sparse.
        # Better: compute M = U @ V.T @ W? No.
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # We can check on a random set of indices or all indices.
        # All indices are small (6*15*10 = 900).
        R = U.shape[1]
        # Compute full tensor
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # This is 900 * R operations.
        # If R=28, 25200 ops. Fast.
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_calc, T, atol=1e-9)

    def snap_to_half_int(A):
        return np.round(A * 2) / 2.0

    def try_rank(R, max_iters=1000, time_limit=10):
        start_time = time.time()
        np.random.seed(42)
        # Initialize
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Normalize to avoid scaling issues
        U = U / np.linalg.norm(U, axis=0, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        
        best_err = float('inf')
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: U = T_mat @ V.T @ W.T / (V.T @ W.T)
            # T_mat is (n*m, m*p, p*n)
            # We want to minimize || U V W^T - T ||
            # For fixed V, W:
            # U_{a,r} = sum_{b,c} T_{a,b,c} V_{b,r} W_{c,r} / sum_{b,c} V_{b,r}^2 W_{c,r}^2
            
            # Compute denominator
            # V.T @ W.T is (R, R)? No.
            # sum_{b,c} V_{b,r} W_{c,r} is not element-wise.
            # The term is sum_{b,c} T_{a,b,c} V_{b,r} W_{c,r}.
            # This is (T * (V_r \otimes W_r)).sum(axis=(1,2))
            
            # Efficient update:
            # Let T_reshaped = T
            # For each r:
            # U[:,r] = (T * (V[:,r,None] * W[:,r][None,:])).sum(axis=(1,2))
            # Denom = (V[:,r] * W[:,r]).sum() ? No.
            # The model is sum_r U_{a,r} V_{b,r} W_{c,r}.
            # This is not a simple matmul.
            # It's a CP decomposition.
            # ALS for CP:
            # U[:,r] = (T_1 * V[:,r] * W[:,r]).sum(axis=(1,2)) / (V[:,r]**2 * W[:,r]**2).sum()
            # Where T_1 is T unfolded along mode 1.
            
            T_1 = T # (n*m, m*p, p*n)
            
            for r in range(R):
                # Update U[:,r]
                numerator = (T_1 * (V[:,r, None] * W[:,r][None, :])).sum(axis=(1, 2))
                denom = (V[:,r]**2 * W[:,r]**2).sum()
                if denom > 1e-9:
                    U[:,r] = numerator / denom
                
                # Update V[:,r]
                numerator = (T_1 * (U[:,r, None] * W[:,r][None, :])).sum(axis=(0, 2))
                denom = (U[:,r]**2 * W[:,r]**2).sum()
                if denom > 1e-9:
                    V[:,r] = numerator / denom
                
                # Update W[:,r]
                numerator = (T_1 * (U[:,r, None] * V[:,r, None])).sum(axis=(0, 1))
                denom = (U[:,r]**2 * V[:,r]**2).sum()
                if denom > 1e-9:
                    W[:,r] = numerator / denom
            
            # Calculate error
            # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            # err = np.linalg.norm(T_calc - T)
            # To save time, check only non-zeros of T?
            # Or just every 10 iterations.
            if it % 10 == 0:
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_calc - T)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-6:
                    break
        
        # Snap to half-integers
        U_s = snap_to_half_int(best_U)
        V_s = snap_to_half_int(best_V)
        W_s = snap_to_half_int(best_W)
        
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s
        
        # Try integer snap
        U_i = np.round(best_U)
        V_i = np.round(best_V)
        W_i = np.round(best_W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
            
        return None

    # Try ranks from 20 up to 29
    for R in range(20, 30):
        result = try_rank(R)
        if result:
            U, V, W = result
            # Convert to list of lists for output
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank = rank_schoolbook
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
