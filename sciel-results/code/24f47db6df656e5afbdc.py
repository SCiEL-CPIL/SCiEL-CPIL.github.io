import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all 1s
    # We want U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    
    # Precompute indices for faster access in loops if needed, 
    # but numpy vectorization is better.
    
    def try_rank(r, max_iters=500, restarts=20, tol=1e-8):
        # Try to find a decomposition of rank r
        for restart in range(restarts):
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Gradient descent
            lr = 0.01
            for it in range(max_iters):
                # Compute S[i, j, k]
                # S_ijk = sum_c U_im+j, c * V_jp+k, c * W_kn+i, c
                # This is a tensor contraction.
                # We can compute this efficiently.
                # For each c, U[:, c] (nm x 1), V[:, c] (mp x 1), W[:, c] (pn x 1)
                # This forms a rank-1 tensor in (i,j,k) space?
                # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                S = np.zeros((n, m, p))
                for c in range(r):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    
                    # T_c[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
                    # This is not a standard outer product because of index coupling.
                    # We can loop over i, j, k for small dimensions.
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                S[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
                                
                E = 1.0 - S
                J = np.sum(E**2)
                
                if J < tol:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    S_snap = np.zeros((n, m, p))
                    for c in range(r):
                        u_c = U_snap[:, c].reshape(n, m)
                        v_c = V_snap[:, c].reshape(m, p)
                        w_c = W_snap[:, c].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    S_snap[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
                                    
                    if np.allclose(S_snap, 1.0, atol=1e-9):
                        # Convert to list of lists
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for c in range(r):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = E[i, j, k]
                                # dJ/du_ijk = -2 * e * v_jk * w_ki
                                grad_U[i*m+j, c] -= 2 * e * v_c[j, k] * w_c[k, i]
                                # dJ/dv_jk = -2 * e * u_ij * w_ki
                                grad_V[j*p+k, c] -= 2 * e * u_c[i, j] * w_c[k, i]
                                # dJ/dw_ki = -2 * e * u_ij * v_jk
                                grad_W[k*n+i, c] -= 2 * e * u_c[i, j] * v_c[j, k]
                                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: decay learning rate
                if it % 50 == 0:
                    lr *= 0.9
                    
        return None

    # Try ranks from 17 down to 10
    start_time = time.time()
    best_result = None
    
    for r in range(17, 9, -1):
        if time.time() - start_time > 120:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
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
        
    return best_result
