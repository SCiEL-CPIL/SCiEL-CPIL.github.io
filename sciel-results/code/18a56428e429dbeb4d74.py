import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank decomposition using iterative linear solves
    # We attempt ranks from 40 down to 30
    best_solution = None
    
    for R in range(40, 29, -2):
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        for _ in range(50):
            # Update W
            for i in range(n):
                for k in range(p):
                    # Solve M @ W_ik = 1
                    # M is m x R
                    M = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    # Least squares
                    W_ik, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_ik
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Solve N @ U_ij = 1
                    # N is p x R
                    N = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            N[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    U_ij, _, _, _ = np.linalg.lstsq(N, np.ones(p), rcond=None)
                    U[i * m + j, :] = U_ij
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    # Solve P @ V_jk = 1
                    # P is n x R
                    P = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            P[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    V_jk, _, _, _ = np.linalg.lstsq(P, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_jk
            
            # Check residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        residual += (val - 1.0) ** 2
            
            if residual < 1e-6:
                # Try rounding to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded solution
                res_round = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_round[i * m + j, r] * V_round[j * p + k, r] * W_round[k * n + i, r]
                            res_round += (val - 1.0) ** 2
                
                if res_round < 1e-5:
                    best_solution = (U_round, V_round, W_round)
                    break
        
        if best_solution:
            break
            
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
