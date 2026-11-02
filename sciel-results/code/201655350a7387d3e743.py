import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor
    # T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if valid (i,j,k), else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    deadline = time.time() + 120  # 120 second budget
    
    # Try ranks from 49 down
    for rank in range(rank_schoolbook - 1, max(10, rank_schoolbook - 15), -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Optimize
            lr = 0.01
            best_residual = np.inf
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_approx
                M = np.einsum('ai,ci->aci', U, W)  # (n*m, p*n, rank)
                T_approx = np.einsum('aci,bi->abc', M, V)  # (n*m, m*p, p*n)
                
                R = T - T_approx
                residual = np.sum(R**2)
                
                if residual < best_residual:
                    best_residual = residual
                
                if residual < 1e-10:
                    break
                
                # Gradients
                # dR/dU[a,i] = -sum_{b,c} R[a,b,c] * V[b,i] * W[c,i]
                grad_U = -np.einsum('abc,bi,ci->ai', R, V, W)
                grad_V = -np.einsum('abc,ai,ci->bi', R, U, W)
                grad_W = -np.einsum('abc,ai,bi->ci', R, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if residual > best_residual * 1.1:
                    lr *= 0.9
                else:
                    lr = min(lr * 1.05, 0.1)
            
            if best_residual < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                M_r = np.einsum('ai,ci->aci', U_rounded, W_rounded)
                T_approx_r = np.einsum('aci,bi->abc', M_r, V_rounded)
                if np.allclose(T_approx_r, T, atol=1e-9):
                    # Convert to lists
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        
    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
