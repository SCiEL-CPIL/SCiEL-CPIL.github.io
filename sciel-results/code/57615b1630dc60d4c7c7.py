import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds deadline
    safety_margin = 10  # seconds to return before deadline
    
    # Start with the rank-8 direct-sum decomposition (each column is a 3x3x1 mat-vec)
    rank = p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k in range(p):
        r = k
        for i in range(n):
            for j in range(m):
                U[i*m + j, r] = 1
        for j in range(m):
            V[j*p + k, r] = 1
        for i in range(n):
            W[k*n + i, r] = 1
    
    def verify(U_mat, V_mat, W_mat):
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        T = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T, target, atol=1e-9)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank using random search
    # Theoretically, rank can be lower than p=8
    # For 3x3x8, the bilinear complexity is at most 8 (direct sum of 8 scalar products)
    # Can we do better? Let's try random searches for lower rank
    
    for target_rank in range(1, p):
        if time.time() > deadline - safety_margin:
            break
            
        # Try multiple random initializations
        num_attempts = 50
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline - safety_margin:
                break
                
            # Random initialization
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Use gradient descent to minimize ||U⊗V⊗W - T||^2
            # Since T has 0/1 entries, we can use this as a loss
            target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = 1.0
            
            lr = 0.01
            for it in range(100):
                if time.time() > deadline - safety_margin:
                    break
                    
                T_cur = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand)
                diff = T_cur - target
                loss = np.sum(diff**2)
                
                if loss < 1e-12:
                    found = True
                    break
                
                # Gradients
                # dL/dU = diff contracted with V and W
                dU = np.einsum('ijk,jr,kr->ir', diff, V_rand, W_rand)
                dV = np.einsum('ijk,ir,kr->jr', diff, U_rand, W_rand)
                dW = np.einsum('ijk,ir,jr->kr', diff, U_rand, V_rand)
                
                U_rand -= lr * dU
                V_rand -= lr * dV
                W_rand -= lr * dW
            
            if found:
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                best_rank = target_rank
                break
        
        if found:
            break
    
    # Verify final result
    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to original rank-8
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
