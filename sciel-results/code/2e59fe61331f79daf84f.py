import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 30  # Lower bound
    max_rank_to_try = 40
    deadline = time.time() + 100  # Leave 20s margin
    
    # Function to check if a decomposition is valid
    def check_decomposition(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # We can vectorize this check
        # Flatten U, V, W to 1D for easier indexing? No, use einsum.
        # U_flat: (n*m*rank)
        # V_flat: (m*p*rank)
        # W_flat: (p*n*rank)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W index is k*n+i, so W_3d[k, i, r]
        # np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check if T is all 1s
        return np.allclose(T, 1.0, atol=1e-9)

    # Function to try to find a decomposition of a given rank
    def find_decomposition(rank, max_iter=1000):
        # Initialize randomly
        # Scale factors to help convergence
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent explosion
        # We want U*V*W ~ 1.
        # Let's scale U, V, W by 1/rank^(1/3)
        scale = 1.0 / (rank ** (1.0/3.0))
        U *= scale
        V *= scale
        W *= scale
        
        # Use gradient descent on the error
        # Error = sum (T - 1)^2
        # This is a non-convex problem.
        # We will use a simple random restart strategy.
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iter in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute T
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            err = np.sum((T - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                # If error is very small, we might have a solution
                if err < 1e-8:
                    # Try to snap to integers/half-integers
                    # This is heuristic
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if check_decomposition(U_snap, V_snap, W_snap, rank):
                        return U_snap, V_snap, W_snap
                    # If snapping fails, keep floating point for now, 
                    # but the task requires integer/half-integer.
                    # We might need to refine.
            
            # Compute gradients
            # dErr/dU[i,j,r] = 2 * (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
            # Sum over k
            grad_U = 2 * np.einsum('ijk,jkr,kir->ijr', (T - 1.0), V_3d, W_3d)
            grad_V = 2 * np.einsum('ijk,ijr,kir->jkr', (T - 1.0), U_3d, W_3d)
            grad_W = 2 * np.einsum('ijk,ijr,jkr->kir', (T - 1.0), U_3d, V_3d)
            
            # Reshape gradients to match U, V, W
            grad_U = grad_U.reshape(n * m, rank)
            grad_V = grad_V.reshape(m * p, rank)
            grad_W = grad_W.reshape(p * n, rank)
            
            # Update
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Random restart if stuck
            if iter % 100 == 0 and err > 1e-4:
                U = np.random.randn(n * m, rank) * scale
                V = np.random.randn(m * p, rank) * scale
                W = np.random.randn(p * n, rank) * scale
        
        # After loop, check best
        if best_err < 1e-8:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            if check_decomposition(U_snap, V_snap, W_snap, rank):
                return U_snap, V_snap, W_snap
        
        return None

    # Try to find a decomposition starting from the lower bound
    for rank in range(target_rank, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        result = find_decomposition(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
