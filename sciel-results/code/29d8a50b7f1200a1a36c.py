import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    best_rank = n * m * p
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    
    U_rows = np.array([i*m + j for i in range(n) for j in range(m) for k in range(p)])
    V_rows = np.array([j*p + k for i in range(n) for j in range(m) for k in range(p)])
    W_rows = np.array([k*n + i for i in range(n) for j in range(m) for k in range(p)])
    
    np.random.seed(42)
    
    def compute_residual_and_grads(U, V, W):
        U_sel = U[U_rows, :]
        V_sel = V[V_rows, :]
        W_sel = W[W_rows, :]
        vals = np.einsum('ta,ta,ta->t', U_sel, V_sel, W_sel)
        diff = vals - 1.0
        res = np.sum(diff**2)
        
        dU = 2.0 * diff[:, None] * V_sel * W_sel
        dV = 2.0 * diff[:, None] * U_sel * W_sel
        dW = 2.0 * diff[:, None] * U_sel * V_sel
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        np.add.at(grad_U, U_rows, dU)
        np.add.at(grad_V, V_rows, dV)
        np.add.at(grad_W, W_rows, dW)
        
        return res, grad_U, grad_V, grad_W
    
    for rank in range(25, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 15
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            scale = 0.3
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            lr = 0.003
            prev_res, _, _, _ = compute_residual_and_grads(U, V, W)
            
            for iteration in range(400):
                if time.time() > deadline - 20:
                    break
                
                res, gU, gV, gW = compute_residual_and_grads(U, V, W)
                
                U_new = U - lr * gU
                V_new = V - lr * gV
                W_new = W - lr * gW
                
                new_res, _, _, _ = compute_residual_and_grads(U_new, V_new, W_new)
                
                if new_res < 1e-10:
                    U, V, W = U_new, V_new, W_new
                    found = True
                    break
                
                if new_res < prev_res:
                    U, V, W = U_new, V_new, W_new
                    prev_res = new_res
                    lr = min(lr * 1.1, 0.02)
                else:
                    lr *= 0.5
                
                if lr < 1e-8:
                    break
            
            if found:
                break
        
        if found:
            best_rank = rank
            best_result = (U, V, W)
            break
    
    U_arr, V_arr, W_arr = best_result
    U_sel = U_arr[U_rows, :]
    V_sel = V_arr[V_rows, :]
    W_sel = W_arr[W_rows, :]
    vals = np.einsum('ta,ta,ta->t', U_sel, V_sel, W_sel)
    res = np.sum((vals - 1.0)**2)
    
    if res > 1e-6:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

