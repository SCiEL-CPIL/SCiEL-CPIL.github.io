import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_solution(U, V, W):
        """Vectorized check of the tensor decomposition."""
        D = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(D, T, atol=1e-6)
    
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    lower_bound = max(n*p, m*p, n*m)
    
    # Try ranks from lower bound up to best_rank
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 40:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            U_try = np.random.randn(nm, target_rank) * 0.3
            V_try = np.random.randn(mp, target_rank) * 0.3
            W_try = np.random.randn(pn, target_rank) * 0.3
            
            # Local search using gradient descent on the residual
            best_residual = np.inf
            U_best, V_best, W_best = U_try.copy(), V_try.copy(), W_try.copy()
            for it in range(400):
                if time.time() > deadline - 20:
                    break
                
                D = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                
                residual = np.sum((D - T)**2)
                if residual < best_residual:
                    best_residual = residual
                    U_best, V_best, W_best = U_try.copy(), V_try.copy(), W_try.copy()
                
                if residual < 1e-10:
                    break
                
                # Compute gradients
                diff = D - T
                
                grad_U = np.einsum('abc,br,cr->ar', diff, V_try, W_try)
                grad_V = np.einsum('abc,ar,cr->br', diff, U_try, W_try)
                grad_W = np.einsum('abc,ar,br->cr', diff, U_try, V_try)
                
                # Adaptive learning rate based on gradient norm
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm < 1e-12:
                    break
                lr = 0.05 / max(grad_norm, 1e-6)
                
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            # Check if we found a good solution
            D = np.einsum('ar,br,cr->abc', U_best, V_best, W_best)
            
            if np.allclose(D, T, atol=1e-6):
                # Round to half-integers and verify
                U_h = to_half_int(U_best)
                V_h = to_half_int(V_best)
                W_h = to_half_int(W_best)
                
                D_h = np.einsum('ar,br,cr->abc', U_h, V_h, W_h)
                if np.allclose(D_h, T, atol=1e-6):
                    best_rank = target_rank
                    U, V, W = U_h, V_h, W_h
                    break
        
        if best_rank == target_rank:
            break
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
