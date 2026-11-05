"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_school = n * m * p  # 72
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def check_solution(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def solve_for_W(U, V):
        W = np.zeros((p*n, U.shape[1]))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = U[i*m+j, :] * V[j*p+k, :]
                    norm_sq = np.dot(c, c)
                    if norm_sq > 1e-14:
                        W[k*n+i, :] = c / norm_sq
        return W
    
    def solve_for_U(V, W):
        U = np.zeros((n*m, V.shape[1]))
        R = V.shape[1]
        for i in range(n):
            for j in range(m):
                D = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])  # p×R
                try:
                    U[i*m+j, :] = np.linalg.lstsq(D.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        return U
    
    def solve_for_V(U, W):
        V = np.zeros((m*p, U.shape[1]))
        R = U.shape[1]
        for j in range(m):
            for k in range(p):
                E = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])  # n×R
                try:
                    V[j*p+k, :] = np.linalg.lstsq(E.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        return V
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    for target_rank in range(rank_school - 1, 65, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(2000 + trial * 7)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                W = solve_for_W(U, V)
                U = solve_for_U(V, W)
                V = solve_for_V(U, W)
            
            if check_solution(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

