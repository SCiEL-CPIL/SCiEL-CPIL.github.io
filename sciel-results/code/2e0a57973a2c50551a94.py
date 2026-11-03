# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = 2, 3, 6
    
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    T = np.zeros((N_U, N_V, N_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def solve_als(r, max_iter=100, seed=0):
        np.random.seed(seed)
        U = np.random.randn(N_U, r)
        V = np.random.randn(N_V, r)
        W = np.random.randn(N_W, r)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = float('inf')
        reg = 1e-8
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Solve U
            for u in range(N_U):
                i = u // m
                j = u % m
                X_sub = np.zeros((p, r))
                y_sub = np.zeros(p)
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    X_sub[k, :] = V[v, :] * W[w, :]
                    y_sub[k] = 1.0
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    U[u, :] = np.linalg.solve(A, b)
                except:
                    U[u, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Solve V
            for v in range(N_V):
                j = v // p
                k = v % p
                X_sub = np.zeros((n, r))
                y_sub = np.zeros(n)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    X_sub[i, :] = U[u, :] * W[w, :]
                    y_sub[i] = 1.0
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    V[v, :] = np.linalg.solve(A, b)
                except:
                    V[v, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Solve W
            for w in range(N_W):
                k = w // n
                i = w % n
                X_sub = np.zeros((m, r))
                y_sub = np.zeros(m)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    X_sub[j, :] = U[u, :] * V[v, :]
                    y_sub[j] = 1.0
                A = X_sub.T @ X_sub + reg * np.eye(r)
                b = X_sub.T @ y_sub
                try:
                    W[w, :] = np.linalg.solve(A, b)
                except:
                    W[w, :] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            R = np.einsum('uc,vc,wc->uvw', U, V, W)
            err = np.sum((R - T)**2)
            
            if err < 1e-10:
                break
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        return U, V, W, err

    def verify_and_round(U, V, W, r):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        R = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        if np.allclose(R, T, atol=1e-8):
            return U_r, V_r, W_r, True
        return None, None, None, False

    best_rank = n * m * p
    U_sb = np.zeros((N_U, best_rank))
    V_sb = np.zeros((N_V, best_rank))
    W_sb = np.zeros((N_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Search for lower rank
    # Try ranks from 18 to 35
    for r in range(18, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        found = False
        # Try a few restarts
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            U, V, W, err = solve_als(r, max_iter=100, seed=seed)
            if err < 1e-8:
                U_r, V_r, W_r, valid = verify_and_round(U, V, W, r)
                if valid:
                    best_rank = r
                    best_decomp = (U_r, V_r, W_r)
                    found = True
                    break
        
        # If we found a low rank, we might want to continue checking lower ranks?
        # The loop goes up, so it will check higher ranks.
        # But we want the MINIMUM.
        # If we found rank 24, rank 23 is lower.
        # So we should NOT break. We should continue the loop?
        # But the loop is `for r in range(18, best_rank)`.
        # If we update `best_rank` to 24, the range is still fixed to `range(18, 36)`.
        # So it will check 25, 26... which are worse.
        # We should iterate downwards?
        # Or just collect the minimum found.
        
    return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
# EVOLVE-BLOCK-END
