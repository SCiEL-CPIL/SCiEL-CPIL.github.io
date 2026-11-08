"""Find rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    start_time = time.time()
    deadline = start_time + 125
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        U3 = np.array(U).reshape(n, m, -1)
        V3 = np.array(V).reshape(m, p, -1)
        W3 = np.array(W).reshape(p, n, -1)
        r = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1.0
        return np.max(np.abs(r)) < tol
    
    def try_rank(rank, num_trials=50, max_steps=4000):
        for trial in range(num_trials):
            if time.time() > deadline - 3:
                return None
            
            np.random.seed(9000 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            mU = np.zeros_like(U); vU = np.zeros_like(U)
            mV = np.zeros_like(V); vV = np.zeros_like(V)
            mW = np.zeros_like(W); vW = np.zeros_like(W)
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            lr = 0.1
            
            for step in range(max_steps):
                if time.time() > deadline - 2:
                    return None
                
                r = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1.0
                L = np.sum(r**2)
                if L < 1e-14:
                    break
                
                dU3 = np.einsum('ijk,jkc,kic->ijc', r, V3, W3)
                dV3 = np.einsum('ijk,ijc,kic->jkc', r, U3, W3)
                dW3 = np.einsum('ijk,ijc,jkc->kic', r, U3, V3)
                
                dU = dU3.reshape(n*m, rank)
                dV = dV3.reshape(m*p, rank)
                dW = dW3.reshape(p*n, rank)
                
                s = step + 1
                mU = b1*mU + (1-b1)*dU; vU = b2*vU + (1-b2)*dU**2
                U -= lr * (mU/(1-b1**s)) / (np.sqrt(vU/(1-b2**s)) + eps)
                
                mV = b1*mV + (1-b1)*dV; vV = b2*vV + (1-b2)*dV**2
                V -= lr * (mV/(1-b1**s)) / (np.sqrt(vV/(1-b2**s)) + eps)
                
                mW = b1*mW + (1-b1)*dW; vW = b2*vW + (1-b2)*dW**2
                W -= lr * (mW/(1-b1**s)) / (np.sqrt(vW/(1-b2**s)) + eps)
                
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
            
            L = np.sum((np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1.0)**2)
            if L < 1e-10:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    for rank in range(15, rank_sb + 1):
        if time.time() > deadline - 10:
            break
        result = try_rank(rank, num_trials=50, max_steps=4000)
        if result is not None:
            return result
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
