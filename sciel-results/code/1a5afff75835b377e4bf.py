import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Target tensor is all ones
    # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
    
    # Strategy: Use known decompositions for smaller all-ones tensors
    # J_3,3,3 rank 23
    # J_3,3,4 rank 27 (hypothetical)
    # Total rank 50
    
    # Since I don't have the explicit coefficients for J_3,3,3 rank 23 hardcoded,
    # I will attempt a search for J_3,3,7 directly with rank 50, 51, 52.
    # Or use a simple heuristic.
    
    # Fallback: Schoolbook rank 63
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # We'll try ranks 50 to 60
    for target_rank in range(50, 61):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.05
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Compute gradient of sum (sum_r ... - 1)^2
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            grad = 2 * (val - 1.0)
                            
                            for r in range(target_rank):
                                dU[i * m + j, r] += grad * V[j * p + k, r] * W[k * n + i, r]
                                dV[j * p + k, r] += grad * U[i * m + j, r] * W[k * n + i, r]
                                dW[k * n + i, r] += grad * U[i * m + j, r] * V[j * p + k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if step % 20 == 0:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    if max_err < 1e-6:
                        # Polish
                        lr_fine = 0.001
                        for step_fine in range(100):
                            dU = np.zeros_like(U)
                            dV = np.zeros_like(V)
                            dW = np.zeros_like(W)
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = 0.0
                                        for r in range(target_rank):
                                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                        grad = 2 * (val - 1.0)
                                        for r in range(target_rank):
                                            dU[i * m + j, r] += grad * V[j * p + k, r] * W[k * n + i, r]
                                            dV[j * p + k, r] += grad * U[i * m + j, r] * W[k * n + i, r]
                                            dW[k * n + i, r] += grad * U[i * m + j, r] * V[j * p + k, r]
                            U -= lr_fine * dU
                            V -= lr_fine * dV
                            W -= lr_fine * dW
                            
                            max_err = 0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = 0.0
                                        for r in range(target_rank):
                                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                        err = abs(val - 1.0)
                                        if err > max_err:
                                            max_err = err
                            if max_err < 1e-8:
                                break
                                
                        if max_err < 1e-8:
                            best_U = U
                            best_V = V
                            best_W = W
                            best_rank = target_rank
                            found = True
                            break
                        else:
                            break
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
