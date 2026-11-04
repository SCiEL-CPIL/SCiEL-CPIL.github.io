import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    rank_school = n * m * p
    U_fb = np.zeros((n*m, rank_school))
    V_fb = np.zeros((m*p, rank_school))
    W_fb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    np.random.seed(42)
    
    for rank in range(12, min(rank_school+1, 50)):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            scale = 0.5
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # Adam optimizer
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t = 0
            lr = 0.02
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                t += 1
                
                u = U.reshape(n, m, rank)
                v = V.reshape(m, p, rank)
                w = W.reshape(p, n, rank)
                
                uv = u[:, :, None, :] * v[None, :, :, :]
                prod = np.einsum('ijka,ika->ijk', uv, w.T)
                residual = 1.0 - prod
                
                grad_U = -np.einsum('ijk,jka,ika->ija', residual, v, w.T)
                grad_V = -np.einsum('ijk,ija,ika->jka', residual, u, w.T)
                grad_W = -np.einsum('ijk,ija,jka->kia', residual, u, v)
                
                grad_U = grad_U.reshape(n*m, rank)
                grad_V = grad_V.reshape(m*p, rank)
                grad_W = grad_W.reshape(p*n, rank)
                
                mU = beta1*mU + (1-beta1)*grad_U
                vU = beta2*vU + (1-beta2)*grad_U**2
                mU_hat = mU/(1-beta1**t)
                vU_hat = vU/(1-beta2**t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1*mV + (1-beta1)*grad_V
                vV = beta2*vV + (1-beta2)*grad_V**2
                mV_hat = mV/(1-beta1**t)
                vV_hat = vV/(1-beta2**t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1*mW + (1-beta1)*grad_W
                vW = beta2*vW + (1-beta2)*grad_W**2
                mW_hat = mW/(1-beta1**t)
                vW_hat = vW/(1-beta2**t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                if np.max(np.abs(residual)) < 1e-10:
                    found = True
                    break
            
            if found:
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
