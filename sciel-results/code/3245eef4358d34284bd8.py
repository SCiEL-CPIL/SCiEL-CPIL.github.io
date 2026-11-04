import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
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
    
    def check_solution_fast(U, V, W):
        """Vectorized check of the tensor decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Build the full tensor: T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # But indices are flattened: U row = i*m+j, V row = j*p+k, W row = k*n+i
        # Constraint: sum_r W[k*n+i, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2)*delta(j,j2)*delta(k2,j)
        
        # For efficiency, check all (i2,j2,k2) for fixed (i,j,k)
        # Actually, let's check the full tensor equality
        # The multiplication tensor: for each (i,j), C[i,k] = sum_j' A[i,j']*B[j',k]
        # So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise? No...
        
        # Let me re-derive. The bilinear form: given A (n×m) and B (m×p), C = A*B (n×p)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In flattened form: A_flat[i*m+j], B_flat[j*p+k], C_flat[k*n+i]
        # The tensor T satisfies: T[A_idx, B_idx, C_idx] = 1 if A_idx=i*m+j, B_idx=j*p+k, C_idx=k*n+i for some i,j,k, else 0
        # Wait, that's not quite right. The tensor decomposition is:
        # C_flat[c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # where a = i*m+j, b = j'*p+k', c = k*n+i'
        # And this should equal sum_j A[i,j]*B[j,k] when i'=i, j'=j, k'=k, else 0
        # So: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
        
        # Let's verify by computing all required values
        # For each (i', j', k'), compute sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i', r] for all (i,j,k)
        # This is expensive. Let's use a smarter check.
        
        # Check: for each (i,j,k), the "diagonal" constraint:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # For off-diagonal: should be 0
        
        # Efficient check: compute the matrix M = W^T @ (U ⊙ V) where ⊙ is some contraction
        # Actually, let's just do it directly but vectorized
        
        # Create index arrays
        i_idx = np.arange(n*m) // m
        j_idx = np.arange(n*m) % m
        
        i2_idx = np.arange(m*p) // p
        j2_idx = np.arange(m*p) % p
        
        k_idx = np.arange(p*n) // n
        i3_idx = np.arange(p*n) % n
        
        # For each rank component r, the contribution is outer product of U[:,r], V[:,r], W[:,r]
        # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] where T is the multiplication tensor
        
        # Let's build the target tensor T as a 3D array
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposed tensor
        # D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is: D = sum_r np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        # More efficiently: D = np.einsum('ar,br,cr->abc', U, V, W)
        
        D = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.allclose(D, T, atol=1e-6)
    
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    # Try structured decomposition based on column grouping
    # Group the 9 columns of B into 3 groups of 3
    # For each group g, we compute C[:, 3g:3g+3] = A @ B[:, 3g:3g+3]
    # This is a 2x3 @ 3x3 = 2x3 multiplication
    
    # For 2x3 @ 3x3, try to find a low-rank decomposition
    # Naive rank = 2*3*3 = 18
    # Let's try to find a rank-9 or lower decomposition for each subproblem
    
    # Actually, let's try a global approach with better search
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    lower_bound = max(n*p, m*p, n*m)
    
    # Try ranks from lower bound up to best_rank
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 40:
            break
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            U_try = np.random.randn(n*m, target_rank) * 0.3
            V_try = np.random.randn(m*p, target_rank) * 0.3
            W_try = np.random.randn(p*n, target_rank) * 0.3
            
            # Local search using gradient descent on the residual
            best_residual = np.inf
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                
                D = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                T = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i*m+j, j*p+k, k*n+i] = 1.0
                
                residual = np.sum((D - T)**2)
                if residual < best_residual:
                    best_residual = residual
                    U_best, V_best, W_best = U_try.copy(), V_try.copy(), W_try.copy()
                
                if residual < 1e-10:
                    break
                
                # Compute gradients
                # d/dU[a,r] = sum_{b,c} (D[a,b,c]-T[a,b,c]) * V[b,r] * W[c,r]
                # d/dV[b,r] = sum_{a,c} (D[a,b,c]-T[a,b,c]) * U[a,r] * W[c,r]
                # d/dW[c,r] = sum_{a,b} (D[a,b,c]-T[a,b,c]) * U[a,r] * V[b,r]
                
                diff = D - T
                
                grad_U = np.einsum('abc,br,cr->ar', diff, V_try, W_try)
                grad_V = np.einsum('abc,ar,cr->br', diff, U_try, W_try)
                grad_W = np.einsum('abc,ar,br->cr', diff, U_try, V_try)
                
                # Simple gradient step with adaptive learning rate
                lr = 0.01
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            # Check if we found a good solution
            D = np.einsum('ar,br,cr->abc', U_best, V_best, W_best)
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
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
