"""Matrix multiplication 2x5 x 5x6 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Precompute the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that U, V, W exactly decompose the multiplication tensor."""
        # pred[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(pred - T))
        return error < tol, error
    
    def try_rank(r, seed=42, max_iters=2000, lr=0.005):
        """Try to find a rank-r decomposition using gradient descent with momentum."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Momentum terms
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        # Precompute for efficiency
        # We'll track best error found
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute prediction
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            diff = pred - T
            
            error = np.sum(diff * diff)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-10:
                U, V, W = best_U, best_V, best_W
                valid, err = verify_decomposition(U, V, W)
                if valid:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    for mat in [U_list, V_list, W_list]:
                        for i in range(len(mat)):
                            for j in range(len(mat[0])):
                                val = mat[i][j]
                                if abs(val - round(val)) < 1e-4:
                                    mat[i][j] = float(round(val))
                                elif abs(2*val - round(2*val)) < 1e-4:
                                    mat[i][j] = float(round(2*val) / 2)
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Compute gradients
            # dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            temp_U = np.einsum('abc,cr->abr', diff, W)
            dU = 2.0 * np.einsum('abr,br->ar', temp_U, V)
            
            # dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            dV = 2.0 * np.einsum('abr,ar->br', temp_U, U)
            
            # dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU
            mV = beta1 * mV + (1 - beta1) * dV
            mW = beta1 * mW + (1 - beta1) * dW
            
            U -= lr * mU / (np.sqrt(mU**2 * beta2) + eps)
            V -= lr * mV / (np.sqrt(mV**2 * beta2) + eps)
            W -= lr * mW / (np.sqrt(mW**2 * beta2) + eps)
        
        # Check if best solution is valid
        valid, err = verify_decomposition(best_U, best_V, best_W)
        if valid:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[0])):
                        val = mat[i][j]
                        if abs(val - round(val)) < 1e-4:
                            mat[i][j] = float(round(val))
                        elif abs(2*val - round(2*val)) < 1e-4:
                            mat[i][j] = float(round(2*val) / 2)
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try decreasing ranks with multiple seeds
    best_result = None
    for r in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 10:
            break
        
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5:
                break
            
            result = try_rank(r, seed=seed, max_iters=1500, lr=0.005)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
