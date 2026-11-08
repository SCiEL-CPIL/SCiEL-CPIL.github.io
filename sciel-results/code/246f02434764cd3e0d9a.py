import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    schoolbook_rank = n * m * p  # 48
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 otherwise
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_tensor(U_mat, V_mat, W_mat):
        """Compute tensor from factors using einsum for speed"""
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
    
    def objective_and_grads(U_mat, V_mat, W_mat):
        """Compute objective and gradients"""
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T
        obj = np.sum(diff**2)
        # Gradients
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return obj, 2*dU, 2*dV, 2*dW
    
    # Try gradient descent for various ranks from low to high
    for rank in range(10, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            U_mat = np.random.randn(n*m, rank) * 0.3
            V_mat = np.random.randn(m*p, rank) * 0.3
            W_mat = np.random.randn(p*n, rank) * 0.3
            
            # Gradient descent
            lr = 0.05
            obj, _, _, _ = objective_and_grads(U_mat, V_mat, W_mat)
            best_obj = obj
            stall_count = 0
            max_iter = 300
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                
                U_new = U_mat - lr * dU
                V_new = V_mat - lr * dV
                W_new = W_mat - lr * dW
                
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_new)
                
                if obj_new < best_obj - 1e-10:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj = obj_new
                    stall_count = 0
                    lr *= 0.998
                else:
                    stall_count += 1
                    lr *= 0.5
                    if lr < 1e-10 or stall_count > 80:
                        break
                
                if best_obj < 1e-10:
                    break
            
            if best_obj < 1e-6:
                # Polish
                lr = 1e-4
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                    U_mat -= lr * dU
                    V_mat -= lr * dV
                    W_mat -= lr * dW
                    lr *= 0.99
                    if obj < 1e-12:
                        break
                
                if objective_and_grads(U_mat, V_mat, W_mat)[0] < 1e-6:
                    best_rank = rank
                    best_U = U_mat.copy()
                    best_V = V_mat.copy()
                    best_W = W_mat.copy()
                    found = True
                    break
        
        if found:
            break
    
    # Convert to required format (list of lists)
    rank = best_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(best_U[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(best_V[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(best_W[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}
