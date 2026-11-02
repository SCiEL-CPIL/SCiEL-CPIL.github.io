"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def schoolbook():
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

    def verify_fast(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    def snap_to_valid(X):
        # Snap to nearest 0.5
        return np.round(X * 2) / 2

    def als_decompose(r, max_iters=100):
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # M[k, c] = V[j*p + k, c] * W[k*n + i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve M u = 1
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[idx, :] = u
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    # M[i, c] = U[i*m + j, c] * W[k*n + i, c]
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[idx, :] = v
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[idx, :] = w
                    except np.linalg.LinAlgError:
                        pass
        
        # Snap to half-integers
        U_snapped = snap_to_valid(U)
        V_snapped = snap_to_valid(V)
        W_snapped = snap_to_valid(W)
        
        # Convert to lists
        U_list = U_snapped.tolist()
        V_list = V_snapped.tolist()
        W_list = W_snapped.tolist()
        
        return U_list, V_list, W_list

    best_result = schoolbook()
    best_rank = n * m * p

    # Lower bound is max(n*m, m*p, p*n) = 28
    lower_bound = max(n * m, m * p, p * n)
    
    # We try ranks from lower_bound up to best_rank
    # But we prioritize lower ranks
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
            
        # More restarts for lower ranks
        restarts = 20 if r < 35 else 5
        
        found = False
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            U, V, W = als_decompose(r, max_iters=50)
            
            if verify_integer(U, V, W) and verify_fast(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                found = True
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
