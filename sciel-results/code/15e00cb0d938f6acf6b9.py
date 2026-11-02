import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor()
    
    def get_trivial():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def check_decomposition(U, V, W, tol=1e-8):
        rank = U.shape[1]
        reconstructed = np.zeros((n*m, m*p, p*n))
        for t in range(rank):
            reconstructed += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        return np.max(np.abs(reconstructed - T)) < tol
    
    def als_decompose(r, max_iters=150, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.3
        V = np.random.randn(m*p, r) * 0.3
        W = np.random.randn(p*n, r) * 0.3
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            M = np.zeros((m*p*p*n, r))
            for t in range(r):
                M[:, t] = np.kron(V[:, t], W[:, t])
            
            for a in range(n*m):
                rhs = T[a].flatten()
                U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Update V
            M = np.zeros((n*m*p*n, r))
            for t in range(r):
                M[:, t] = np.kron(U[:, t], W[:, t])
            
            for b in range(m*p):
                rhs = T[:, b, :].flatten()
                V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            # Update W
            M = np.zeros((n*m*m*p, r))
            for t in range(r):
                M[:, t] = np.kron(U[:, t], V[:, t])
            
            for c in range(p*n):
                rhs = T[:, :, c].flatten()
                W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            
            if it % 20 == 0 and check_decomposition(U, V, W, tol=1e-5):
                break
        
        # Try rounding to half-integers
        for scale in [2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if check_decomposition(U_r, V_r, W_r, tol=1e-10):
                return U_r, V_r, W_r
        
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_decomposition(U_i, V_i, W_i, tol=1e-10):
            return U_i, V_i, W_i
        
        return None
    
    best = get_trivial()
    
    # Try to find lower rank
    found_better = False
    for r in range(44, 35, -1):
        if time.time() > deadline or found_better:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            result = als_decompose(r, max_iters=80, seed=seed)
            if result is not None:
                U, V, W = result
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found_better = True
                break
    
    return best
