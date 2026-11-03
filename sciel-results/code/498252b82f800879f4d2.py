import numpy as np
import scipy.linalg as la

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Tensor T is 3D array of shape (n, m, p)
    # T[i, j, k] = 1 for all i, j, k
    T = np.ones((n, m, p), dtype=float)
    
    # We want to find U, V, W such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i, j, k]
    
    # Try increasing ranks
    # Lower bound is max(nm, mp, pn) = 16
    for rank in range(16, 49):
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # IHT steps
        for it in range(100):
            # Compute current tensor
            # Reshape U, V, W to use einsum
            # U is (n, m, rank), V is (m, p, rank), W is (p, n, rank)
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # T_current[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
            # Note: W index is (k, i).
            T_current = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
            
            # Residual
            R = T - T_current
            
            # Update U, V, W using gradient descent or IHT
            # For simplicity, use a few steps of gradient descent on the factors
            # Gradient of T w.r.t U:
            # dT/dU[i, j, r] = V[j, k, r] * W[k, i, r] (summed over k)
            grad_U = np.einsum('jkr,ki r->ijr', V_3d, W_3d)
            grad_V = np.einsum('ijr,ki r->jkr', U_3d, W_3d)
            grad_W = np.einsum('ijr,jkr->ki r', U_3d, V_3d)
            
            # Note: grad_W index is (k, i, r). W_3d is (p, n, rank).
            # grad_W shape is (p, n, rank).
            
            # Update
            lr = 0.01
            U -= lr * grad_U.reshape(nm, rank)
            V -= lr * grad_V.reshape(mp, rank)
            W -= lr * grad_W.reshape(pn, rank)
            
            # Check convergence
            if np.linalg.norm(R) < 1e-6:
                # Snap to nearest half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                T_final = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
                if np.allclose(T_final, T, atol=1e-6):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # Fallback to seed
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
