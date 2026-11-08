import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    best_result = None
    best_rank = target_rank
    
    # Schoolbook decomposition
    U_sb = np.zeros((n*m, target_rank))
    V_sb = np.zeros((m*p, target_rank))
    W_sb = np.zeros((p*n, target_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    start_time = time.time()
    deadline = start_time + 110
    
    for rank in range(12, target_rank):
        if time.time() > deadline:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n*m, rank) * 1.0
            V = np.random.randn(m*p, rank) * 1.0
            W = np.random.randn(p*n, rank) * 1.0
            
            def loss_and_grad(params):
                U_flat, V_flat, W_flat = np.split(params, [n*m*rank, n*m*rank + m*p*rank])
                U_c = U_flat.reshape(n, m, rank)
                V_c = V_flat.reshape(m, p, rank)
                W_c = W_flat.reshape(p, n, rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_c, V_c, W_c)
                loss = np.sum((T - 1.0)**2)
                
                dT = 2 * (T - 1.0)
                
                dU = np.einsum('ijk,jkc,kic->ijc', dT, V_c, W_c)
                dV = np.einsum('ijk,ijc,kic->jkc', dT, U_c, W_c)
                dW = np.einsum('ijk,ijc,jkc->kic', dT, U_c, V_c)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, jac=True, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6})
            except Exception:
                continue
            
            if res.fun < 1e-4:
                U_sol = res.x[:n*m*rank].reshape(n*m, rank)
                V_sol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                U_r_snap = U_snap.reshape(n, m, rank)
                V_r_snap = V_snap.reshape(m, p, rank)
                W_r_snap = W_snap.reshape(p, n, rank)
                T_snap = np.einsum('ijc,jkc,kic->ijk', U_r_snap, V_r_snap, W_r_snap)
                
                if np.allclose(T_snap, 1.0, atol=1e-9):
                    best_rank = rank
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    if rank <= 14:
                        return best_result

    return best_result
