import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Check if T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k"""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_error_vectorized(U, V, W):
        """Compute MSE using vectorized operations"""
        n, m, p = DIMENSIONS
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j, :]  # shape (r,)
                v_row = V[j * p:(j+1)*p, :]  # shape (p, r)
                for k in range(p):
                    v_k = v_row[k, :]  # shape (r,)
                    w_k = W[k * n + i, :]  # shape (r,)
                    s = np.dot(u_row, v_k * w_k)
                    err += (s - 1.0) ** 2
        return err
    
    def try_rank_gradient(r, max_time=30):
        """Try to find a rank-r decomposition using gradient descent"""
        start = time.time()
        np.random.seed(42 + r)
        
        for attempt in range(5):
            if time.time() > start + max_time:
                break
            
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            lr = 0.01
            best_err = compute_error_vectorized(U, V, W)
            
            for step in range(2000):
                if time.time() > start + max_time:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        u_row = U[i * m + j, :]
                        for k in range(p):
                            v_k = V[j * p + k, :]
                            w_k = W[k * n + i, :]
                            s = np.dot(u_row, v_k * w_k)
                            err_grad = 2 * (s - 1.0)
                            dU[i * m + j, :] += err_grad * (v_k * w_k)
                            dV[j * p + k, :] += err_grad * (u_row * w_k)
                            dW[k * n + i, :] += err_grad * (u_row * v_k)
                
                # Gradient descent
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.999
                
                if step % 100 == 0:
                    current_err = compute_error_vectorized(U, V, W)
                    if current_err < best_err:
                        best_err = current_err
                    if best_err < 1e-8:
                        break
            
            if best_err < 1e-8:
                if verify(U, V, W, tol=1e-7):
                    return U, V, W, r
        
        return None
    
    # Try ranks from 56 down to 50
    for r in range(56, 49, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank_gradient(r, max_time=20)
        if result is not None:
            U, V, W, rk = result
            best_rank = rk
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
