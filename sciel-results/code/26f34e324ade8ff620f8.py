import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    def time_left():
        return time_limit - (time.time() - start_time)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def compute_grads(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        E = 1.0 - T
        
        # Gradients
        dU_r = -2.0 * np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
        dV_r = -2.0 * np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
        dW_r = -2.0 * np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
        
        return dU_r.reshape(-1, R), dV_r.reshape(-1, R), dW_r.reshape(-1, R)

    def adam_optimize(U, V, W, R, steps=500, lr=0.01):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        while t < steps:
            if time_left() < 5.0:
                break
                
            dU, dV, dW = compute_grads(U, V, W, R)
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.sum((1.0 - T) ** 2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_error < 1e-10:
                break
                
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            m_U = beta1 * m_U + (1 - beta1) * dU
            v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
            U -= lr_t * m_U / (np.sqrt(v_U) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * dV
            v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
            V -= lr_t * m_V / (np.sqrt(v_V) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
            W -= lr_t * m_W / (np.sqrt(v_W) + eps)
            
        return best_U, best_V, best_W, best_error

    # Search ranks from low to high
    for R in range(20, 48):
        if time_left() < 10.0:
            break
            
        max_restarts = 5 if time_left() > 30.0 else 2
        
        for restart in range(max_restarts):
            if time_left() < 5.0:
                break
                
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            best_U, best_V, best_W, err = adam_optimize(U, V, W, R, steps=500, lr=0.01)
            
            if err < 1e-8:
                if verify(best_U, best_V, best_W):
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
