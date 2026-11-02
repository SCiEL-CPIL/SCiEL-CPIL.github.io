"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Build target tensor T[u,v,w]
    # T[i*m+j, j'*p+k, k'*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomp(U, V, W):
        T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.max(np.abs(T_comp - T))
    
    # Correct schoolbook rank-48
    rank48 = n * m * p  # 48
    U48 = np.zeros((n*m, rank48))
    V48 = np.zeros((m*p, rank48))
    W48 = np.zeros((p*n, rank48))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U48[i*m+j, idx] = 1.0
                V48[j*p+k, idx] = 1.0
                W48[k*n+i, idx] = 1.0
                idx += 1
    
    best_rank = rank48
    best_decomp = (U48, V48, W48)
    
    # Try to find lower rank via gradient descent
    for target_rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(1000 + target_rank * 100 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Adam-like optimizer
            lr = 0.05
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            t_step = 0
            
            for step in range(500):
                if time.time() > deadline - 3:
                    break
                
                t_step += 1
                T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
                diff = T_comp - T
                
                dU = np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = np.einsum('uvw,ur,wr->vr', diff, U, W)
                dW = np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                # Adam update
                mU = beta1*mU + (1-beta1)*dU
                vU = beta2*vU + (1-beta2)*dU**2
                mU_hat = mU/(1-beta1**t_step)
                vU_hat = vU/(1-beta2**t_step)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = beta1*mV + (1-beta1)*dV
                vV = beta2*vV + (1-beta2)*dV**2
                mV_hat = mV/(1-beta1**t_step)
                vV_hat = vV/(1-beta2**t_step)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = beta1*mW + (1-beta1)*dW
                vW = beta2*vW + (1-beta2)*dW**2
                mW_hat = mW/(1-beta1**t_step)
                vW_hat = vW/(1-beta2**t_step)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                if step % 50 == 0:
                    err = np.max(np.abs(T_comp - T))
                    if err < 1e-5:
                        found = True
                        break
            
            if found:
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
        if found:
            break
    
    U, V, W = best_decomp
    U_list = [[float(U[i,j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i,j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i,j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
