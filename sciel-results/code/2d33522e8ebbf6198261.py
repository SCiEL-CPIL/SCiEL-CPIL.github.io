"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
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

    start_time = time.time()
    deadline = start_time + 140.0

    # Search for lower rank
    for R in range(40, 19, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts
        for restart in range(100):
            if time.time() > deadline - 5.0:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Alternating Least Squares
            for it in range(100):
                # Update W
                M_W = np.zeros((p * n, R))
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            M_W[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
                for idx in range(p * n):
                    norm_sq = np.dot(M_W[idx, :], M_W[idx, :])
                    if norm_sq > 1e-10:
                        W[idx, :] = M_W[idx, :] / norm_sq
                    else:
                        W[idx, :] = 0
                        
                # Update U
                N_U = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            N_U[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
                for idx in range(n * m):
                    norm_sq = np.dot(N_U[idx, :], N_U[idx, :])
                    if norm_sq > 1e-10:
                        U[idx, :] = N_U[idx, :] / norm_sq
                    else:
                        U[idx, :] = 0

                # Update V
                N_V = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            N_V[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                for idx in range(m * p):
                    norm_sq = np.dot(N_V[idx, :], N_V[idx, :])
                    if norm_sq > 1e-10:
                        V[idx, :] = N_V[idx, :] / norm_sq
                    else:
                        V[idx, :] = 0

                # Check residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-6:
                    # Try to round to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    max_err_int = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                                err = abs(val - 1.0)
                                if err > max_err_int:
                                    max_err_int = err
                    if max_err_int < 1e-9:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }

    return schoolbook()

# EVOLVE-BLOCK-END
