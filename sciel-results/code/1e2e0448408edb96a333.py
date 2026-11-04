import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Fallback schoolbook decomposition
    def get_schoolbook():
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

    def verify(U, V, W, r):
        T_comp = np.zeros((n, m, p))
        for c in range(r):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if u_val == 0: continue
                    for k in range(p):
                        T_comp[i, j, k] += u_val * v_col[j * p + k] * w_col[k * n + i]
        return np.allclose(T_comp, 1.0, atol=1e-9)

    def try_find_decomposition(r, num_tries=200):
        for _ in range(num_tries):
            if time.time() > deadline:
                return None
            
            # Use small integers for U and V to increase chance of rational W
            # Range [-2, 2]
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # Solve for W
            W_sol = np.zeros((p * n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A_ik[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    try:
                        w_vec, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        if np.linalg.norm(A_ik @ w_vec - np.ones(m)) > 1e-9:
                            valid = False
                            break
                        W_sol[k * n + i, :] = w_vec
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check if W is close to half-integers
                W_snap = np.round(W_sol * 2) / 2
                # Check U, V are already integers, so they are valid
                if verify(U, V, W_snap, r):
                    return U, V, W_snap
        return None

    # Try ranks from 44 down to 40
    # If we find 44, score is 45/44 = 1.022
    # If we find 40, score is 45/40 = 1.125
    for r in range(44, 39, -1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(r, num_tries=100)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return get_schoolbook()
