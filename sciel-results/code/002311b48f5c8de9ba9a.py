import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    schoolbook_rank = n * m * p
    
    # Try ranks from a reasonable lower bound to schoolbook
    # For M_{4,5,5}, lower bound is at least max(n*m, m*p, n*p) = 25
    # But likely much higher. Let's try from 50 to 100.
    
    for R in range(50, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + R * 1000 + restart)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            best_error = float('inf')
            best_U_local = None
            best_V_local = None
            best_W_local = None
            
            for it in range(80):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                A_u = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        W_idx = np.arange(p) * n + i
                        V_slice = V[j * p:(j + 1) * p, :]
                        W_slice = W[W_idx, :]
                        A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
                
                norms_sq = np.sum(A_u * A_u, axis=1)
                norms_sq[norms_sq == 0] = 1e-10
                U = A_u / norms_sq[:, np.newaxis]
                
                # Solve for V
                A_v = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        U_idx = np.arange(n) * m + j
                        W_idx = k * n + np.arange(n)
                        U_slice = U[U_idx, :]
                        W_slice = W[W_idx, :]
                        A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
                
                norms_sq_v = np.sum(A_v * A_v, axis=1)
                norms_sq_v[norms_sq_v == 0] = 1e-10
                V = A_v / norms_sq_v[:, np.newaxis]
                
                # Solve for W
                A_w = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        U_idx = i * m + np.arange(m)
                        V_idx = np.arange(m) * p + k
                        U_slice = U[U_idx, :]
                        V_slice = V[V_idx, :]
                        A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
                
                norms_sq_w = np.sum(A_w * A_w, axis=1)
                norms_sq_w[norms_sq_w == 0] = 1e-10
                W = A_w / norms_sq_w[:, np.newaxis]
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < best_error:
                    best_error = err
                    best_U_local = U.copy()
                    best_V_local = V.copy()
                    best_W_local = W.copy()
                
                if best_error < 1e-10:
                    break
            
            if best_error < 1e-8:
                U_int = np.round(best_U_local * 2) / 2
                V_int = np.round(best_V_local * 2) / 2
                W_int = np.round(best_W_local * 2) / 2
                
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                is_exact = False
                                break
                        if not is_exact:
                            break
                    if not is_exact:
                        break
                
                if is_exact:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
