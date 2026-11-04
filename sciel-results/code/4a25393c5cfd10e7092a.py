"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from 30 to 41
    for target_rank in range(30, rank_sb):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(12345 + target_rank * 7 + restart)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)
            
            # Alternating least squares style optimization
            for it in range(2000):
                if time.time() > deadline:
                    break
                
                # Update U: for each i, U[i,:] = argmin ||T[i,:,:] - U[i,:] @ (V*W)^T||^2
                # T[i,:,:] = sum_c U[i,c] * V[:,c] * W[:,c]
                # Let M = V * W (element-wise), shape (m*p, p*n, target_rank)
                # Then T[i,:,:] = sum_c U[i,c] * M[:,:,c]
                # This is a linear system: T[i,:,:].reshape(-1) = U[i,:] @ M.reshape(m*p*p*n, target_rank).T
                # But M is 3D, so we need to handle it carefully
                
                # Actually, for fixed V and W, the problem for U[i,:] is:
                # minimize || sum_c U[i,c] * V[:,c] * W[:,c] - T[i,:,:] ||^2
                # This is linear in U[i,:]
                # Let A[i] be a matrix where A[i][j*k, c] = V[j,c] * W[k,c] for j in range(m*p), k in range(p*n)
                # Then T[i,:,:].reshape(-1) = U[i,:] @ A[i].T
                # So U[i,:] = T[i,:,:].reshape(-1) @ pinv(A[i].T)
                
                # This is expensive. Let's use gradient descent instead.
                
                T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                diff = T_recon - T
                loss = np.sum(diff ** 2)
                
                if loss < 1e-8:
                    break
                
                dU = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
                dV = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
                dW = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
                
                # Adaptive learning rate
                lr = 0.001
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check and round
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            loss = np.sum((T_recon - T) ** 2)
            
            if loss < 1e-5:
                # Try rounding to nearest half-integer
                for scale in [2, 4, 8]:  # Try different precisions
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    
                    T_recon_r = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                    loss_r = np.sum((T_recon_r - T) ** 2)
                    
                    if loss_r < 1e-10:
                        best_result = (U_r, V_r, W_r)
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
