"""Program searching for low-rank exact tensor decomposition of 2x3 by 3x8 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 8]
SEED = 42

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook baseline
    base_rank = n * m * p
    U0 = [[0] * base_rank for _ in range(n * m)]
    V0 = [[0] * base_rank for _ in range(m * p)]
    W0 = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best = {"U": U0, "V": V0, "W": W0, "rank": base_rank}
    
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose T."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-10, rtol=1e-10)
    
    def tensor_residual(U, V, W, T):
        """Compute ||T - sum_c U_c V_c W_c||_F^2"""
        computed = np.einsum('ic,jc,kc->ijk', U, V, W)
        diff = T - computed
        return np.sum(diff * diff)
    
    def gradient_descent(U, V, W, T, lr=0.01, max_iter=500, tol=1e-12):
        """Simple gradient descent on tensor rank decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        prev_loss = tensor_residual(U, V, W, T)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if prev_loss < tol:
                break
            
            # Gradients:
            # dL/dU[i,c] = -2 * sum_{j,k} T[i,j,k]*V[j,c]*W[k,c] + 2*sum_{j,k} computed[i,j,k]*V[j,c]*W[k,c]
            # = -2 * (T * V[:,:,c] * W[:,:,c]) + 2 * (computed * V[:,:,c] * W[:,:,c])
            # where * is element-wise over the other two dims
            
            computed = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            # grad_U[i,c] = 2 * (computed - T) contracted with V[j,c]*W[k,c] over j,k
            # grad_U = 2 * einsum('ijk,jc,kc->ic', computed - T, V, W)
            diff = computed - T
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_loss = tensor_residual(U, V, W, T)
            if new_loss < prev_loss * 0.999:
                lr = min(lr * 1.05, 0.1)
            else:
                lr = max(lr * 0.95, 1e-6)
            prev_loss = new_loss
        
        return U, V, W, prev_loss
    
    def try_rank(r, rng, max_restarts=3):
        """Try to find rank-r decomposition using gradient descent."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            scale = 1.0 / np.sqrt(r)
            U = rng.normal(size=(n * m, r)) * scale
            V = rng.normal(size=(m * p, r)) * scale
            W = rng.normal(size=(p * n, r)) * scale
            
            U, V, W, loss = gradient_descent(U, V, W, T, lr=0.05, max_iter=300, tol=1e-14)
            
            if loss < 1e-10:
                # Found exact decomposition
                return U, V, W
        
        return None
    
    # Search for lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Try ranks from 24 up to 47
    min_possible = max(n * m, m * p, p * n)
    
    rng = np.random.RandomState(SEED)
    
    for r in range(min_possible, base_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(r, rng, max_restarts=2)
        if result is not None:
            U, V, W = result
            # Verify
            if verify(U, V, W, T):
                # Convert to list format with rounding to half-integers if close
                U_list = [[float(round(x * 2) / 2) for x in row] for row in U]
                V_list = [[float(round(x * 2) / 2) for x in row] for row in V]
                W_list = [[float(round(x * 2) / 2) for x in row] for row in W]
                
                # Verify after rounding
                if verify(U_list, V_list, W_list, T):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Return without rounding
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook decomposition
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
