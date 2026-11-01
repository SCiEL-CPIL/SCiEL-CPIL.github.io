import numpy as np
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                for j in range(m):
                    T[i, j, k] = 1.0

    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try ranks from 12 to best_rank - 1
    for r in range(12, best_rank):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        for restart in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS
            for it in range(100):
                # Update U
                for i in range(n):
                    for j in range(m):
                        # M is 5 x r
                        M = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # T[i, j, :] is length p
                        # Solve M^T u = T[i, j, :]
                        # u is length r
                        try:
                            u = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                            U[i * m + j, :] = u
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # N is 2 x r
                        N = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                N[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        # T[:, j, k] is length n
                        try:
                            v = np.linalg.lstsq(N.T, T[:, j, k], rcond=None)[0]
                            V[j * p + k, :] = v
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # P is 2 x r
                        P = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                P[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        # T[i, :, k] is length m
                        try:
                            w = np.linalg.lstsq(P.T, T[i, :, k], rcond=None)[0]
                            W[k * n + i, :] = w
                        except:
                            pass

            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - T[i, j, k]) ** 2
            
            if error < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped
                error_snap = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_snap[i * m + j, c] * V_snap[j * p + k, c] * W_snap[k * n + i, c]
                            error_snap += (val - T[i, j, k]) ** 2
                
                if error_snap < 1e-9:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
