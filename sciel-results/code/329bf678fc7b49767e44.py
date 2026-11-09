import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_vals.ravel()
    j_flat = j_vals.ravel()
    k_flat = k_vals.ravel()
    
    U_idx = i_flat * m + j_flat
    V_idx = j_flat * p + k_flat
    W_idx = k_flat * n + i_flat
    
    T = len(i_flat)  # 30
    
    for rank in range(15, 31):
        if time.time() > deadline - 15:
            break
        
        for trial in range(40):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 7919 + rank * 31 + 12345)
            
            # Initialize with structure: some entries to 1
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Random initialization
            U += np.random.randn(n*m, rank) * 0.3
            V += np.random.randn(m*p, rank) * 0.3
            W += np.random.randn(p*n, rank) * 0.3
            
            best_res = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for iteration in range(500):
                if time.time() > deadline - 3:
                    break
                
                U_sel = U[U_idx]
                V_sel = V[V_idx]
                W_sel = W[W_idx]
                
                val = np.einsum('tr,tr,tr->t', U_sel, V_sel, W_sel)
                err = val - 1.0
                res = float(np.sum(err * err))
                
                if res < best_res:
                    best_res = res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if res < 1e-12:
                    break
                
                contrib_U = 2 * err[:, None] * V_sel * W_sel
                contrib_V = 2 * err[:, None] * U_sel * W_sel
                contrib_W = 2 * err[:, None] * U_sel * V_sel
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                np.add.at(grad_U, U_idx, contrib_U)
                np.add.at(grad_V, V_idx, contrib_V)
                np.add.at(grad_W, W_idx, contrib_W)
                
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm < 1e-14:
                    break
                
                lr = 0.05 / (1 + grad_norm)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if best_res < 1e-8:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                U_sel = U_r[U_idx]
                V_sel = V_r[V_idx]
                W_sel = W_r[W_idx]
                val = np.einsum('tr,tr,tr->t', U_sel, V_sel, W_sel)
                if float(np.max(np.abs(val - 1))) < 1e-4:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
