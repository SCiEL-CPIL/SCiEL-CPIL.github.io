"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank 72 as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try ALS to find rank < 72 decomposition
    def try_rank_als(r, max_iters=100, seed=None):
        if time.time() > deadline - 10:
            return None
        
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Fix U, V, solve for W
            W_new = np.zeros_like(W)
            for i in range(n):
                for kp in range(p):
                    M = np.zeros((m*m*p, r))
                    t = np.zeros(m*m*p)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for c in range(r):
                                    M[idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        W_new[kp*n+i, :] = w
                    except Exception:
                        pass
            W = W_new
            
            # Fix U, W, solve for V
            V_new = np.zeros_like(V)
            for jp in range(m):
                for k in range(p):
                    M = np.zeros((n*m*p, r))
                    t = np.zeros(n*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for kp in range(p):
                                for c in range(r):
                                    M[idx, c] = U[i*m+j, c] * W[kp*n+i, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        V_new[jp*p+k, :] = v
                    except Exception:
                        pass
            V = V_new
            
            # Fix V, W, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((m*p*p, r))
                    t = np.zeros(m*p*p)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for c in range(r):
                                    M[idx, c] = V[jp*p+k, c] * W[kp*n+i, c]
                                t[idx] = 1.0 if (j == jp and k == kp) else 0.0
                                idx += 1
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        U_new[i*m+j, :] = u
                    except Exception:
                        pass
            U = U_new
        
        # Check validity
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (j == jp and k == kp) else 0.0
                            val = sum(U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+i, c] for c in range(r))
                            err = abs(val - expected)
                            if err > max_err:
                                max_err = err
        
        if max_err < 1e-6:
            U_list = [[round(val * 2) / 2 for val in row] for row in U]
            V_list = [[round(val * 2) / 2 for val in row] for row in V]
            W_list = [[round(val * 2) / 2 for val in row] for row in W]
            
            max_err2 = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (j == jp and k == kp) else 0.0
                                val = sum(U_list[i*m+j][c] * V_list[jp*p+k][c] * W_list[kp*n+i][c] for c in range(r))
                                err = abs(val - expected)
                                if err > max_err2:
                                    max_err2 = err
            
            if max_err2 < 1e-8:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 71 down to 64
    for r in range(71, 63, -1):
        if time.time() > deadline - 30:
            break
        for seed in [42, 123]:
            if time.time() > deadline - 20:
                break
            result = try_rank_als(r, max_iters=50, seed=seed)
            if result is not None:
                return result
    
    return schoolbook()
# EVOLVE-BLOCK-END
