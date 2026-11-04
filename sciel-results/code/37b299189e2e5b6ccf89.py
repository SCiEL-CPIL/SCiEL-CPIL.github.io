import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Rank 27 Fallback
    U27 = np.zeros((9, 27))
    V27 = np.zeros((9, 27))
    W27 = np.zeros((9, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1
                V27[j*p+k, col] = 1
                W27[k*n+i, col] = 1
                col += 1
        
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the 3x3x3 matrix multiplication tensor."""
        r = U.shape[1]
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    T[a, b, c] = 1.0
        
        R = np.einsum('at,bt,ct->abc', U, V, W)
        
        if not np.allclose(R, T, atol=tol, rtol=0):
            return False
        return True

    def attempt_rank(r, max_iter=1000, seed=42):
        """Attempt to find a rank r decomposition using ALS."""
        np.random.seed(seed)
        U = np.random.randn(9, r)
        V = np.random.randn(9, r)
        W = np.random.randn(9, r)
        
        # Reshape T for ALS
        # T[a, b, c] = sum_t U[a, t] V[b, t] W[c, t]
        # Fix V, W -> U[a, t] = sum_{b,c} T[a, b, c] V[b, t] W[c, t] / (V[b,t]^2 W[c,t]^2 + eps)? No, this is not linear in U if we don't fix the others.
        # Actually, for fixed V, W, the equation is linear in U?
        # T[a, b, c] = U[a, :] dot (V[b, :] * W[c, :])
        # So U[a, t] can be solved via least squares if we treat the "right hand side" appropriately.
        # Let's reshape.
        # T_reshaped = np.einsum('abc->ab c', T) -> shape (9, 9, 9)
        # We want U @ (V * W)^T ? No.
        # T[a, b, c] = sum_t U[a, t] (V[b, t] * W[c, t])
        # Let M[b, c, t] = V[b, t] * W[c, t]
        # Then T[a, b, c] = sum_t U[a, t] M[b, c, t]
        # This is a linear system for U if we flatten b, c.
        # T_flat[a, idx(b,c)] = U[a, t] M_flat[idx(b,c), t]
        # So U[a, :] = T_flat[a, :] @ M_flat.T @ inv(M_flat @ M_flat.T)
        
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    T[a, b, c] = 1.0
                    
        T_flat = T.reshape(9, 81)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            M_VW = np.einsum('bt,ct->bct', V, W) # Shape (9, 9, r)
            M_VW_flat = M_VW.reshape(81, r)
            # Solve U[a, :] M_VW_flat.T = T_flat[a, :]
            # U = T_flat @ M_VW_flat.T @ inv(M_VW_flat @ M_VW_flat.T)
            try:
                G = M_VW_flat @ M_VW_flat.T
                G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                U = T_flat @ M_VW_flat.T @ G_inv
            except:
                pass
            
            # Update V
            M_UW = np.einsum('at,ct->act', U, W) # Shape (9, 9, r)
            M_UW_flat = M_UW.reshape(81, r)
            try:
                G = M_UW_flat @ M_UW_flat.T
                G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                V = T_flat @ M_UW_flat.T @ G_inv
            except:
                pass
                
            # Update W
            M_UV = np.einsum('at,bt->abt', U, V) # Shape (9, 9, r)
            M_UV_flat = M_UV.reshape(81, r)
            try:
                G = M_UV_flat @ M_UV_flat.T
                G_inv = np.linalg.inv(G + 1e-9 * np.eye(r))
                W = T_flat @ M_UV_flat.T @ G_inv
            except:
                pass
            
            # Check convergence / verify
            if it % 10 == 0:
                R = np.einsum('at,bt,ct->abc', U, V, W)
                if np.allclose(R, T, atol=1e-8, rtol=0):
                    # Round to integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    # Try half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    if verify(U_half, V_half, W_half):
                        return U_half, V_half, W_half
                    
        return None

    # Try Rank 23
    # Use multiple seeds/restarts
    for seed in range(10):
        res = attempt_rank(23, max_iter=500, seed=seed)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to Rank 27
    return {
        "U": U27.tolist(),
        "V": V27.tolist(),
        "W": W27.tolist()
    }
