DIMENSIONS = [2, 2, 4]
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, target_rank, max_iters=5000, seed=42):
    """Try to find a decomposition of the given rank using random search + local refinement."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    # Scale to make products closer to 1
    for r in range(target_rank):
        # Normalize so that the "average" contribution is reasonable
        pass
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute loss: sum over all (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = s - 1.0
                    loss += err * err
                    for r in range(target_rank):
                        dU = err * 2 * V[j*p+k, r] * W[k*n+i, r]
                        dV = err * 2 * U[i*m+j, r] * W[k*n+i, r]
                        dW = err * 2 * U[i*m+j, r] * V[j*p+k, r]
                        grad_U[i*m+j, r] += dU
                        grad_V[j*p+k, r] += dV
                        grad_W[k*n+i, r] += dW
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradient step with adaptive learning rate
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Periodic random restarts within the iteration
        if it % 500 == 0 and it > 0:
            # Small perturbation
            U += 0.01 * rng.randn(*U.shape)
            V += 0.01 * rng.randn(*V.shape)
            W += 0.01 * rng.randn(*W.shape)
    
    if best_loss < 1e-10:
        # Convert to list format
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        # Try to snap to rational values (half-integers)
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
        # Verify after snapping
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
        else:
            # Return unsnapped if snapped fails
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave margin from 150s limit
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # For 2x2x2, rank is 7. For 2x2x4, let's try from 8 upward.
    # Actually, a simple lower bound: the tensor has 16 nonzero entries, 
    # but rank can be less. Let's try from 8.
    
    # First, try a structured approach: two Strassen blocks (rank 14)
    # Then try to find lower ranks via search.
    
    # Try ranks from 8 to 15
    for target_rank in range(8, 16):
        if time.time() > deadline - 10:
            break
        result = _search_rank(n, m, p, target_rank, max_iters=2000, seed=target_rank*1000)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to trivial solution
    return _trivial_solution(n, m, p)
