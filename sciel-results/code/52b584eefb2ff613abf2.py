import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a lower-rank decomposition
    deadline = time.time() + 100  # Leave 20s margin
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try ranks from 47 down to some minimum
    for target_rank in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current decomposition
                # Sum over c of U[:,c] outer V[:,c] outer W[:,c]
                # This is expensive, so we'll compute the residual efficiently
                # Residual = T - sum_c U_c ⊗ V_c ⊗ W_c
                # For gradient, we need d/dU, d/dV, d/dW
                
                # Compute the "matricized" residual
                # R1 = reshape of residual to (n*m, m*p*p*n)
                # But this is huge. Let's use a different approach.
                
                # Actually, for small tensors, we can just compute it directly
                # T is 6x24x16 = 2304 entries, which is manageable
                
                # Compute sum of rank-1 terms
                S = np.zeros_like(T)
                for c in range(target_rank):
                    S += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                
                R = T - S
                
                # Gradient with respect to U: d/dU[a,c] = sum_{b,d} R[a,b,d] * V[b,c] * W[d,c]
                # Similarly for V and W
                dU = np.einsum('abd,bc,dc->ac', R, V, W)
                dV = np.einsum('abd,ac,dc->bc', R, U, W)
                dW = np.einsum('abd,ac,bc->dc', R, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if np.linalg.norm(R) < 1e-6:
                    break
            
            # Check if we found a good decomposition
            S = np.zeros_like(T)
            for c in range(target_rank):
                S += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            R = T - S
            if np.linalg.norm(R) < 1e-4:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                S_rounded = np.zeros_like(T)
                for c in range(target_rank):
                    S_rounded += np.einsum('a,b,c->abc', U_rounded[:,c], V_rounded[:,c], W_rounded[:,c])
                if np.allclose(S_rounded, T, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    # Return the best found
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
