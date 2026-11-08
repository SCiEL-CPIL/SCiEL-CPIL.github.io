import numpy as np
import time

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

    def compute_tensor(U_mat, V_mat, W_mat):
        """Compute tensor from factors using einsum for speed"""
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)

    def objective_and_grads(U_mat, V_mat, W_mat):
        """Compute objective and gradients"""
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T
        obj = np.sum(diff**2)
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return obj, 2*dU, 2*dV, 2*dW

    # Strategy: Partition 2x4x6 into two 2x2x6 subproblems
    # Split the inner dimension m=4 into two parts of size 2
    # Subproblem 1: 2x2x6 (i=0,1; j=0,1; k=0..5)
    # Subproblem 2: 2x2x6 (i=0,1; j=2,3; k=0..5)
    # Each 2x2x6 tensor can potentially be decomposed with rank < 24
    # Known: 2x2x3 has rank 7 (Strassen-like), so 2x2x6 might have rank ~13-14
    
    # Let's try to find a low-rank decomposition for the full tensor
    # but with structure-aware initialization

    # First, try direct optimization with various ranks
    # Use a smarter initialization: start from schoolbook and try to merge

    def try_rank(rank, max_restarts=10):
        """Try to find a rank decomposition for the given rank"""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None, None, None, False
            
            # Initialize with structured random
            U_mat = np.random.randn(n*m, rank) * 0.5
            V_mat = np.random.randn(m*p, rank) * 0.5
            W_mat = np.random.randn(p*n, rank) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.1
            obj, _, _, _ = objective_and_grads(U_mat, V_mat, W_mat)
            best_obj = obj
            stall_count = 0
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None, None, None, False
                
                obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                
                # Adaptive step size
                step_scale = lr / (1 + np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW)))
                
                U_new = U_mat - step_scale * dU
                V_new = V_mat - step_scale * dV
                W_new = W_mat - step_scale * dW
                
                obj_new, _, _, _ = objective_and_grads(U_new, V_new, W_new)
                
                if obj_new < best_obj - 1e-12:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj = obj_new
                    stall_count = 0
                    lr *= 1.01
                else:
                    stall_count += 1
                    lr *= 0.7
                    if lr < 1e-10 or stall_count > 100:
                        break
                
                if best_obj < 1e-8:
                    break
            
            if best_obj < 1e-6:
                # Polish with smaller steps
                lr = 1e-3
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    obj, dU, dV, dW = objective_and_grads(U_mat, V_mat, W_mat)
                    step_scale = lr / (1 + np.max(np.abs(dU)) + np.max(np.abs(dV)) + np.max(np.abs(dW)))
                    U_mat -= step_scale * dU
                    V_mat -= step_scale * dV
                    W_mat -= step_scale * dW
                    lr *= 0.995
                    if obj < 1e-10:
                        break
                
                final_obj = objective_and_grads(U_mat, V_mat, W_mat)[0]
                if final_obj < 1e-6:
                    return U_mat, V_mat, W_mat, True
        
        return None, None, None, False

    # Try ranks from low to high
    # Lower bound: For 2x4x6, the tensor rank is at least max(n*m, m*p, p*n) = max(8,24,12) = 24
    # But we can do better than schoolbook's 48
    # Try ranks: 24, 25, 26, ..., up to 48
    
    for rank in range(24, best_rank + 1):
        if time.time() > deadline:
            break
        
        U_mat, V_mat, W_mat, found = try_rank(rank, max_restarts=5)
        
        if found:
            best_rank = rank
            best_U = U_mat.copy()
            best_V = V_mat.copy()
            best_W = W_mat.copy()
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
