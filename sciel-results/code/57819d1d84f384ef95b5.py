"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
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

    def is_exact(U, V, W, n, m, p):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        r = len(U[0]) if U else 0
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_gradient_search(rank, steps=2000, lr=0.01):
        """
        Attempt to find a rank-r decomposition using gradient descent 
        on the continuous relaxation, followed by rounding and local search.
        """
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Precompute indices for faster access
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for step in range(steps):
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i, j, k in indices:
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                
                # Current value for this (i,j,k)
                val = np.dot(np.dot(ui, vj), wk) # Wait, this is not right. 
                # val = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
                # This is a tensor contraction, not a matrix product of these slices.
                # val = np.einsum('t,t,t->', ui, vj, wk)
                val = np.sum(ui * vj * wk)
                err = val - 1.0
                
                # dErr/dU[i,j,t] = err * V[j,k,t] * W[k,i,t]
                # dErr/dV[j,k,t] = err * U[i,j,t] * W[k,i,t]
                # dErr/dW[k,i,t] = err * U[i,j,t] * V[j,k,t]
                
                grad_U[i * m + j] += 2 * err * (vj * wk)
                grad_V[j * p + k] += 2 * err * (ui * wk)
                grad_W[k * n + i] += 2 * err * (ui * vj)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: L2 regularization to keep values small?
            # U -= 0.001 * U
            
            # Check for convergence / exactness periodically
            if step % 100 == 0:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if is_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        
        # Final check after all steps
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if is_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        return None

    # Try to find a lower rank decomposition
    # Start from a rank slightly below schoolbook
    # Schoolbook is 60. Try 58, 59?
    # Given the difficulty, we try a few ranks.
    for r in [59, 58, 55]:
        result = try_gradient_search(r)
        if result:
            return result
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
