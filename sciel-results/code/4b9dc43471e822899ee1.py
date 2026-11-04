"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition (rank 84) as guaranteed fallback
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

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(R, max_iters=5000, lr=0.01):
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            S = np.einsum('ijr,jkr,kin->ijk', U_r, V_r, W_r)
            err = np.sum((S - 1.0)**2)
            
            if err < 1e-12:
                return U, V, W
                
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            dS = 2.0 * (S - 1.0)
            dU = np.einsum('ijk,jkr,kin->ijr', dS, V_r, W_r)
            dV = np.einsum('ijk,ijr,kin->jkr', dS, U_r, W_r)
            dW = np.einsum('ijk,ijr,jkr->kin', dS, U_r, V_r)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 100 == 0 and err > 1.0:
                lr = lr * 0.9
            elif it % 100 == 0 and err < 0.1:
                lr = lr * 1.1
                
        if best_err < 1e-9:
            return best_U, best_V, best_W
        return None

    for R in range(83, 70, -1):
        if time.time() > deadline - 20:
            break
            
        found = None
        for restart in range(20):
            if time.time() > deadline - 20:
                break
            res = try_rank(R)
            if res is not None:
                U, V, W = res
                U_2 = np.round(U * 2) / 2
                V_2 = np.round(V * 2) / 2
                W_2 = np.round(W * 2) / 2
                
                U_list = U_2.tolist()
                V_list = V_2.tolist()
                W_list = W_2.tolist()
                
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
    return schoolbook()
# EVOLVE-BLOCK-END
