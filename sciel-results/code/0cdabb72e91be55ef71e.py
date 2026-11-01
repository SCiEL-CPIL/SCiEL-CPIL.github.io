import numpy as np
import time
import scipy.optimize as so

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Build target tensor
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def verify(U, V, W):
        """Verify exact decomposition with half-integer entries"""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
        
        # Check exact decomposition
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True
    
    def snap_and_verify(U, V, W):
        """Snap to nearest half-integer and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None
    
    def compute_residual(U, V, W):
        """Compute residuals for the tensor decomposition"""
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return T - T_calc
    
    def als_decompose(rank, max_iter=150, tol=1e-12):
        """Alternating Least Squares decomposition"""
        # Random initialization with reasonable scale
        U = np.random.randn(U_size, rank) * 0.7
        V = np.random.randn(V_size, rank) * 0.7
        W = np.random.randn(W_size, rank) * 0.7
        
        prev_cost = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: For each u, solve linear system
            # T[u, v, w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # Fix V,W, solve for U[u,:]
            for u in range(U_size):
                # Build matrix: for each (v,w) pair, M[(v,w), c] = V[v,c] * W[w,c]
                # b[(v,w)] = T[u, v, w]
                M = np.zeros((V_size * W_size, rank))
                b = np.zeros(V_size * W_size)
                idx = 0
                for v in range(V_size):
                    for w in range(W_size):
                        M[idx, :] = V[v, :] * W[w, :]
                        b[idx] = T[u, v, w]
                        idx += 1
                
                U[u, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            
            # Update V: For each v, solve linear system
            for v in range(V_size):
                M = np.zeros((U_size * W_size, rank))
                b = np.zeros(U_size * W_size)
                idx = 0
                for u in range(U_size):
                    for w in range(W_size):
                        M[idx, :] = U[u, :] * W[w, :]
                        b[idx] = T[u, v, w]
                        idx += 1
                
                V[v, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            
            # Update W: For each w, solve linear system
            for w in range(W_size):
                M = np.zeros((U_size * V_size, rank))
                b = np.zeros(U_size * V_size)
                idx = 0
                for u in range(U_size):
                    for v in range(V_size):
                        M[idx, :] = U[u, :] * V[v, :]
                        b[idx] = T[u, v, w]
                        idx += 1
                
                W[w, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
            
            # Calculate cost
            res = compute_residual(U, V, W)
            cost = np.sum(res ** 2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        # Try to snap to half-integers
        result = snap_and_verify(U, V, W)
        if result is not None:
            return result
        
        # If close, try polish with scipy
        if cost < 1e-6:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals_func(x):
                U_flat = x[:U_size*rank].reshape(U_size, rank)
                V_flat = x[U_size*rank : U_size*rank + V_size*rank].reshape(V_size, rank)
                W_flat = x[U_size*rank + V_size*rank :].reshape(W_size, rank)
                return compute_residual(U_flat, V_flat, W_flat).flatten()
            
            try:
                res_polish = so.least_squares(residuals_func, x0, method='lm', max_nfev=500, ftol=1e-15)
                x_opt = res_polish.x
                U_opt = x_opt[:U_size*rank].reshape(U_size, rank)
                V_opt = x_opt[U_size*rank : U_size*rank + V_size*rank].reshape(V_size, rank)
                W_opt = x_opt[U_size*rank + V_size*rank :].reshape(W_size, rank)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
            except Exception:
                pass
        
        return None
    
    def try_structured(rank):
        """Try structured decomposition based on known patterns"""
        # For 2x3x3, try a pattern based on Strassen-like ideas
        # This is a heuristic initialization
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        
        # Initialize with a structured pattern
        for c in range(rank):
            # Create diverse patterns
            U[:, c] = np.random.choice([-1, 0, 1], size=U_size) * 0.5
            V[:, c] = np.random.choice([-1, 0, 1], size=V_size) * 0.5
            W[:, c] = np.random.choice([-1, 0, 1], size=W_size) * 0.5
        
        # Run ALS from this initialization
        return als_decompose(rank, max_iter=100)
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try ranks from 14 upwards
    for R in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        # Number of restarts based on time available
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        max_restarts = min(50, int(time_left / 2))
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Alternate between random and structured initialization
            if restart % 3 == 0:
                result = try_structured(R)
            else:
                result = als_decompose(R)
            
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # Also try with different random seeds for better coverage
        np.random.seed(restart * 1000)
        for restart2 in range(min(10, max_restarts // 5)):
            if time.time() > deadline:
                break
            result = als_decompose(R)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
