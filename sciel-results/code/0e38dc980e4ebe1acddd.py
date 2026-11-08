import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

def verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = np.sum(np.array(U[i*m+j]) * np.array(V[j*p+k]) * np.array(W[k*n+i]))
                if abs(prod - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140
    
    # Known rank-28 solution as fallback
    def rank_28():
        rank = 28
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        r = 0
        for i in range(2):
            for k in range(5):
                U[i*3+2][r]=1; V[2*5+k][r]=1; W[k*2+i][r]=1; r+=1
        for i in range(2):
            for j in range(2):
                U[i*3+j][r]=1; V[j*5+4][r]=1; W[4*2+i][r]=1; r+=1
        # Strassen for j=0,1 k=0,1
        U[0][r]=1;U[4][r]=1;V[0][r]=1;V[6][r]=1;W[0][r]=1;W[3][r]=1;r+=1
        U[3][r]=1;U[4][r]=1;V[0][r]=1;W[1][r]=1;W[3][r]=-1;r+=1
        U[0][r]=1;V[1][r]=1;V[6][r]=-1;W[2][r]=1;W[3][r]=1;r+=1
        U[4][r]=1;V[5][r]=1;V[0][r]=-1;W[0][r]=1;W[1][r]=1;r+=1
        U[0][r]=1;U[1][r]=1;V[6][r]=1;W[0][r]=-1;W[2][r]=1;r+=1
        U[3][r]=1;U[0][r]=-1;V[0][r]=1;V[1][r]=1;W[3][r]=1;r+=1
        U[1][r]=1;U[4][r]=-1;V[5][r]=1;V[6][r]=1;W[0][r]=1;r+=1
        # Strassen for j=0,1 k=2,3
        U[0][r]=1;U[4][r]=1;V[2][r]=1;V[8][r]=1;W[4][r]=1;W[7][r]=1;r+=1
        U[3][r]=1;U[4][r]=1;V[2][r]=1;W[5][r]=1;W[7][r]=-1;r+=1
        U[0][r]=1;V[3][r]=1;V[8][r]=-1;W[6][r]=1;W[7][r]=1;r+=1
        U[4][r]=1;V[7][r]=1;V[2][r]=-1;W[4][r]=1;W[5][r]=1;r+=1
        U[0][r]=1;U[1][r]=1;V[8][r]=1;W[4][r]=-1;W[6][r]=1;r+=1
        U[3][r]=1;U[0][r]=-1;V[2][r]=1;V[3][r]=1;W[7][r]=1;r+=1
        U[1][r]=1;U[4][r]=-1;V[7][r]=1;V[8][r]=1;W[4][r]=1;r+=1
        return {"U":U,"V":V,"W":W}
    
    best = rank_28()
    
    # Try to find rank 27 or lower using numerical optimization
    for target_rank in [27, 26, 25]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + target_rank + attempt)
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # residual
                            res = 0.0
                            for r in range(target_rank):
                                res += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res -= 1.0
                            for r in range(target_rank):
                                U_grad[i*m+j, r] += res * V[j*p+k, r] * W[k*n+i, r]
                                V_grad[j*p+k, r] += res * U[i*m+j, r] * W[k*n+i, r]
                                W_grad[k*n+i, r] += res * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    # Found a solution, round to integers/half-integers
                    U_list = [[round(x, 1) for x in row] for row in U]
                    V_list = [[round(x, 1) for x in row] for row in V]
                    W_list = [[round(x, 1) for x in row] for row in W]
                    if verify_decomposition(n, m, p, U_list, V_list, W_list, 1e-4):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
            
            if found:
                break
    
    return best
