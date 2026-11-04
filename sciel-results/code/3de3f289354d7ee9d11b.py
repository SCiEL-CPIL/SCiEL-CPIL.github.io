"""Matrix multiplication 3,5,5 - rank-minimal tensor decomposition"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    time_limit = 140.0  # Leave 10 seconds margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions using optimization
    # The tensor we're decomposing has the property that:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use a structured approach based on known matrix multiplication algorithms
    # For small matrices, we can try various rank values and use optimization
    
    def try_rank(rank, max_iter=2000, lr=0.01):
        """Try to find a decomposition of given rank using gradient descent."""
        if time.time() - start_time > time_limit - 5:
            return None
        
        # Initialize with small random values
        np.random.seed(rank * 1000 + 42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Precompute index arrays for efficiency
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            loss = 0.0
            for (i, j, k) in indices:
                ui = U[i*m+j, :]
                vj = V[j*p+k, :]
                wk = W[k*n+i, :]
                
                prod = ui * vj * wk
                target = 1.0
                diff = np.sum(prod) - target
                
                loss += diff * diff
                
                # Gradients
                U_grad[i*m+j, :] += 2 * diff * vj * wk
                V_grad[j*p+k, :] += 2 * diff * ui * wk
                W_grad[k*n+i, :] += 2 * diff * ui * vj
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            if loss < 1e-12:
                break
        
        # Check if we found a valid solution
        final_loss = 0.0
        for (i, j, k) in indices:
            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            final_loss += (s - 1.0) ** 2
        
        if final_loss < 1e-9:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded version
            valid = True
            for (i, j, k) in indices:
                s = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            
            if valid:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
