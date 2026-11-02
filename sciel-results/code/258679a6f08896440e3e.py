"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, rank_s = make_schoolbook()
    best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist(), rank_s)
    
    # Try to find lower rank
    for target_rank in range(20, 24):
        if time.time() > deadline - 15:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            converged = False
            for it in range(300):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        N = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                N[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # U[i*m+j,:] @ N.T = ones(p)
                        if target_rank >= p:
                            U[i * m + j, :] = np.linalg.lstsq(N.T, np.ones(p), rcond=None)[0]
                        else:
                            U[i * m + j, :] = np.linalg.lstsq(N.T, np.ones(p), rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        P = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                P[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        V[j * p + k, :] = np.linalg.lstsq(P, np.ones(n), rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        Q = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                Q[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        W[k * n + i, :] = np.linalg.lstsq(Q, np.ones(m), rcond=None)[0]
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(target_rank))
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-12:
                    converged = True
                    break
            
            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                max_err_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c] for c in range(target_rank))
                            max_err_r = max(max_err_r, abs(val - 1.0))
                
                if max_err_r < 1e-10:
                    best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist(), target_rank)
                    break
                else:
                    best_result = (U.tolist(), V.tolist(), W.tolist(), target_rank)
                    break
        
        if best_result[3] < rank_s:
            break
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END

