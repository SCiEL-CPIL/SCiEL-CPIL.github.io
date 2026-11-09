"""Program to find rank-minimal tensor decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 100
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-6):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def try_rank_als(r, seed, max_iters=800):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r) * 0.3
        V = rng.randn(mp, r) * 0.3
        W = rng.randn(pn, r) * 0.3
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    u, _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    U_new[idx_u, :] = u
            
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    C = np.zeros((n, r))
                    for i in range(n):
                        C[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    v, _, _, _ = np.linalg.lstsq(C.T, np.ones(n), rcond=None)
                    V_new[idx_v, :] = v
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    idx_w = k * n + i
                    D = np.zeros((m, r))
                    for j in range(m):
                        D[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    w, _, _, _ = np.linalg.lstsq(D.T, np.ones(m), rcond=None)
                    W_new[idx_w, :] = w
            
            if verify(U_new, V_new, W_new, tol=1e-8):
                return (U_new, V_new, W_new, r)
            
            alpha = 0.7
            U = alpha * U + (1 - alpha) * U_new
            V = alpha * V + (1 - alpha) * V_new
            W = alpha * W + (1 - alpha) * W_new
        
        return None
    
    # Try ranks in descending order of likelihood
    for r in [85, 80, 75, 70, 65, 60, 55, 50]:
        if time.time() > deadline:
            break
        for seed in range(8):
            if time.time() > deadline:
                break
            result = try_rank_als(r, seed=seed, max_iters=600)
            if result is not None:
                U, V, W, rank = result
                if verify(U, V, W, tol=1e-6):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
        if best_rank < n*m*p:
            break
    
    return best_result
# EVOLVE-BLOCK-END
