import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        strassen_products = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
        ]
        
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    k_global = k1 + k_sub
                    V[j * p + k_global, t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    k_global = k1 + k_sub
                    W[k_global * n + i, t] = coeff
        return U, V, W

    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = 140.0
    
    best_rank = 28
    best_U, best_V, best_W = get_strassen_28()
    
    for target_rank in [27, 26, 25]:
        if time.time() > deadline - 25: break
        
        for restart in range(3):
            if time.time() > deadline - 15: break
            
            np.random.seed(100 + target_rank * 10 + restart)
            
            U_curr = np.random.randn(n*m, target_rank) * 0.1
            V_curr = np.random.randn(m*p, target_rank) * 0.1
            W_curr = np.random.randn(p*n, target_rank) * 0.1
            
            for it in range(80):
                if time.time() > deadline - 12: break
                
                T_calc = np.einsum('at,bt,ct->abc', U_curr, V_curr, W_curr)
                err = T_calc - T
                loss = np.sum(err**2)
                
                if loss < 1e-8:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    break
                
                gU = np.einsum('abc,bt,ct->at', err, V_curr, W_curr)
                gV = np.einsum('abc,at,ct->bt', err, U_curr, W_curr)
                gW = np.einsum('abc,at,bt->ct', err, U_curr, V_curr)
                
                lr = 0.02
                U_curr -= lr * gU
                V_curr -= lr * gV
                W_curr -= lr * gW
            
            if best_rank == target_rank:
                break
    
    if time.time() > deadline - 5:
        U, V, W = best_U, best_V, best_W
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_U, best_V, best_W
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
