import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 140  # safety margin before 150s wall-clock limit
    
    # Target: tensor T[i,j,k] = delta_{i,0} * delta_{j,0} * delta_{k,0} = 1 if i=j=k=0 else 0
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Fallback: naive decomposition of rank n*m*p
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    def _verify(U, V, W, rank):
        """Check if decomposition exactly reproduces the target tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - target) > 1e-9:
                        return False
        return True
    
    # Try increasing ranks starting from small values
    for rank in range(1, 50):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 30 if rank <= 10 else 10
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 1000 + restart * 42)
            
            # Random initialization with small values
            scale = 0.3
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            # Gradient descent to minimize sum_{i,j,k} (sum_r U_ijr V_jkr W_kir - target_ijk)^2
            lr = 0.05
            best_loss = float('inf')
            
            for iteration in range(2000):
                if time.time() > deadline - 15:
                    break
                
                # Compute residuals and gradients
                dU = np.zeros((n*m, rank))
                dV = np.zeros((m*p, rank))
                dW = np.zeros((p*n, rank))
                loss = 0.0
                
                for i in range(n):
                    for j in range(m):
                        target = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                        for k in range(p):
                            target = 1.0 if (i == 0 and j == 0 and k == 0) else 0.0
                            res = 0.0
                            for r in range(rank):
                                res += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            res -= target
                            loss += res * res
                            
                            for r in range(rank):
                                dU[i*m+j, r] += res * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += res * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += res * U[i*m+j, r] * V[j*p+k, r]
                
                # Check convergence
                if loss < 1e-12:
                    found = True
                    break
                
                if loss < best_loss:
                    best_loss = loss
                    U_best = U.copy()
                    V_best = V.copy()
                    W_best = W.copy()
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Reduce learning rate occasionally
                if iteration > 0 and iteration % 500 == 0:
                    lr *= 0.5
            
            if found:
                # Verify with exact check
                U_list = U_best.tolist() if best_loss < 1e-12 else U.tolist()
                V_list = V_best.tolist() if best_loss < 1e-12 else V.tolist()
                W_list = W_best.tolist() if best_loss < 1e-12 else W.tolist()
                
                # Round to half-integers if close
                U_r = np.round(np.array(U_list) * 2) / 2
                V_r = np.round(np.array(V_list) * 2) / 2
                W_r = np.round(np.array(W_list) * 2) / 2
                
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), rank):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                elif _verify(U_list, V_list, W_list, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the naive decomposition
    return {"U": U_fb, "V": V_fb, "W": W_fb}
